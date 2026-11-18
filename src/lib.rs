//! `pyroomacoustics.libroom` re-implemented in Rust.
//!
//! This module is a drop-in replacement of the original pybind11/C++ extension.
//! It exposes the same classes, functions, attributes and overloads, and it
//! produces bit-identical results (see eig.rs for how Eigen's floating point
//! evaluation order is reproduced).

mod eig;
mod fast;
mod geom;
mod mic;
mod pool;
mod room;
mod rir;
mod rng;
mod rt_kernels;
mod tracer;
mod wall;

use numpy::{Element, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::mic::{Histogram2D as HistData, MicData};
use crate::room::{Geometry, IsmResults, Params};
use crate::wall::WallData;

// ---------------------------------------------------------------------------
// conversion helpers
// ---------------------------------------------------------------------------

fn numpy<'py>(py: Python<'py>) -> PyResult<&'py Bound<'py, PyModule>> {
    static NP: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    NP.get_or_try_init(py, || Ok(py.import("numpy")?.unbind()))
        .map(|m| m.bind(py))
}

/// Converts any array-like to a C-contiguous array of the given dtype
/// (like pybind11's forcecast conversion). Returns (shape, data).
fn to_array<'py, T: Element + Copy>(
    obj: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<(Vec<usize>, Vec<T>)> {
    let py = obj.py();
    let np = numpy(py)?;
    let kw = PyDict::new(py);
    kw.set_item("dtype", np.getattr(dtype)?)?;
    let arr = np
        .getattr("ascontiguousarray")?
        .call((obj,), Some(&kw))
        .map_err(|_| PyTypeError::new_err("incompatible function arguments"))?;
    let arr = arr
        .cast::<PyArrayDyn<T>>()
        .map_err(|_| PyTypeError::new_err("incompatible function arguments"))?;
    let ro = arr.readonly();
    let shape = ro.shape().to_vec();
    let data = ro.as_slice().map_err(|e| PyTypeError::new_err(e.to_string()))?.to_vec();
    Ok((shape, data))
}

fn f32s(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<f32>)> {
    to_array::<f32>(obj, "float32")
}

fn type_err() -> PyErr {
    PyTypeError::new_err("incompatible function arguments")
}

/// Fixed size vector (Eigen::Matrix<float, D, 1>)
fn vec_fixed<const D: usize>(obj: &Bound<'_, PyAny>) -> PyResult<[f32; D]> {
    let (shape, data) = f32s(obj)?;
    let ok = match shape.len() {
        1 => shape[0] == D,
        2 => (shape[0] == D && shape[1] == 1) || (shape[0] == 1 && shape[1] == D),
        _ => false,
    };
    if !ok {
        return Err(type_err());
    }
    let mut r = [0.0f32; D];
    r.copy_from_slice(&data);
    Ok(r)
}

/// Dynamic vector (Eigen::VectorXf / ArrayXf)
fn vec_dyn(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let (shape, data) = f32s(obj)?;
    let ok = match shape.len() {
        0 => true,
        1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => false,
    };
    if !ok {
        return Err(type_err());
    }
    Ok(data)
}

/// Matrix with a given number of rows (fixed) and dynamic columns: returns columns.
fn mat_cols<const R: usize>(obj: &Bound<'_, PyAny>) -> PyResult<Vec<[f32; R]>> {
    let (shape, data) = f32s(obj)?;
    let (rows, cols) = match shape.len() {
        1 if R == 1 => (1, shape[0]),
        1 if shape[0] == R => (R, 1),
        2 => (shape[0], shape[1]),
        _ => return Err(type_err()),
    };
    if rows != R {
        return Err(type_err());
    }
    Ok((0..cols)
        .map(|j| {
            let mut c = [0.0f32; R];
            for i in 0..R {
                c[i] = data[i * cols + j];
            }
            c
        })
        .collect())
}

/// Matrix with dynamic rows and fixed number of columns: returns rows.
fn mat_rows<const C: usize>(obj: &Bound<'_, PyAny>) -> PyResult<Vec<[f32; C]>> {
    let (shape, data) = f32s(obj)?;
    let (rows, cols) = match shape.len() {
        1 if shape[0] == C => (1, C),
        2 => (shape[0], shape[1]),
        _ => return Err(type_err()),
    };
    if cols != C {
        return Err(type_err());
    }
    Ok((0..rows)
        .map(|i| {
            let mut r = [0.0f32; C];
            r.copy_from_slice(&data[i * cols..(i + 1) * cols]);
            r
        })
        .collect())
}

/// Dynamic matrix: (rows, cols, row-major data)
fn mat_dyn(obj: &Bound<'_, PyAny>) -> PyResult<(usize, usize, Vec<f32>)> {
    let (shape, data) = f32s(obj)?;
    match shape.len() {
        1 => Ok((shape[0], 1, data)),
        2 => Ok((shape[0], shape[1], data)),
        _ => Err(type_err()),
    }
}

fn out_1d<'py, T: Element + Copy>(py: Python<'py>, v: &[T]) -> Bound<'py, PyArray1<T>> {
    PyArray1::from_slice(py, v)
}

/// Build a Fortran ordered (column-major, like Eigen) 2D array.
fn out_2d_f<'py, T: Element + Copy>(
    py: Python<'py>,
    rows: usize,
    cols: usize,
    f: impl Fn(usize, usize) -> T,
) -> Bound<'py, PyArray2<T>> {
    let arr = unsafe { PyArray2::<T>::new(py, [rows, cols], true) };
    {
        let s = unsafe { arr.as_slice_mut().expect("contiguous") };
        for j in 0..cols {
            for i in 0..rows {
                s[j * rows + i] = f(i, j);
            }
        }
    }
    arr
}

fn readonly<'py, T>(a: Bound<'py, T>) -> PyResult<Bound<'py, PyAny>> {
    let a = a.into_any();
    let kw = PyDict::new(a.py());
    kw.set_item("write", false)?;
    a.call_method("setflags", (), Some(&kw))?;
    Ok(a)
}

/// Writable float32 output buffer of size D (`Eigen::Ref<Vectorf<D>>`)
fn write_out<const D: usize>(obj: &Bound<'_, PyAny>, v: &[f32; D]) -> PyResult<()> {
    let arr = obj.cast::<PyArrayDyn<f32>>().map_err(|_| type_err())?;
    if arr.len() != D || !arr.is_contiguous() {
        return Err(type_err());
    }
    let mut rw = arr.try_readwrite().map_err(|_| type_err())?;
    let s = rw.as_slice_mut().map_err(|_| type_err())?;
    s.copy_from_slice(v);
    Ok(())
}

fn read_out<const D: usize>(obj: &Bound<'_, PyAny>) -> PyResult<[f32; D]> {
    let arr = obj.cast::<PyArrayDyn<f32>>().map_err(|_| type_err())?;
    if arr.len() != D || !arr.is_contiguous() {
        return Err(type_err());
    }
    let ro = arr.readonly();
    let s = ro.as_slice().map_err(|_| type_err())?;
    let mut r = [0.0f32; D];
    r.copy_from_slice(s);
    Ok(r)
}

fn rt_err<E: std::fmt::Debug>(msg: E) -> PyErr {
    let s = format!("{:?}", msg);
    PyRuntimeError::new_err(s)
}

fn is_int(obj: &Bound<'_, PyAny>) -> bool {
    // emulate pybind11 size_t caster: python ints / objects with __index__, not floats
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return false;
    }
    obj.extract::<u64>().is_ok()
}

// ---------------------------------------------------------------------------
// Histogram2D, Hit
// ---------------------------------------------------------------------------

/// Where the data of a Histogram2D lives: either owned, or a live view into
/// one of the histograms of a microphone (pybind11 returns references).
pub enum HistOwner {
    Own(HistData),
    Mic3(Py<PyMic3>, usize),
    Mic2(Py<PyMic2>, usize),
}

#[pyclass(name = "Histogram2D", module = "pyroomacoustics.libroom")]
pub struct PyHistogram2D {
    pub owner: HistOwner,
}

fn stale() -> PyErr {
    PyIndexError::new_err("the microphone histogram referenced by this object no longer exists")
}

impl PyHistogram2D {
    fn with<R>(&self, py: Python<'_>, f: impl FnOnce(&HistData) -> R) -> PyResult<R> {
        match &self.owner {
            HistOwner::Own(h) => Ok(f(h)),
            HistOwner::Mic3(m, i) => m.borrow(py).inner.histograms.get(*i).map(f).ok_or_else(stale),
            HistOwner::Mic2(m, i) => m.borrow(py).inner.histograms.get(*i).map(f).ok_or_else(stale),
        }
    }
    fn with_mut<R>(&mut self, py: Python<'_>, f: impl FnOnce(&mut HistData) -> R) -> PyResult<R> {
        match &mut self.owner {
            HistOwner::Own(h) => Ok(f(h)),
            HistOwner::Mic3(m, i) => m.borrow_mut(py).inner.histograms.get_mut(*i).map(f).ok_or_else(stale),
            HistOwner::Mic2(m, i) => m.borrow_mut(py).inner.histograms.get_mut(*i).map(f).ok_or_else(stale),
        }
    }
}

#[pymethods]
impl PyHistogram2D {
    #[new]
    fn new(r: i32, c: i32) -> Self {
        PyHistogram2D { owner: HistOwner::Own(HistData::new(r.max(0) as usize, c.max(0) as usize)) }
    }
    fn log(&mut self, py: Python<'_>, row: i64, col: i64, val: f32) -> PyResult<()> {
        self.with_mut(py, |h| h.log(row as usize, col as usize, val))
    }
    fn bin(&self, py: Python<'_>, row: i64, col: i64) -> PyResult<f32> {
        self.with(py, |h| {
            let (r, c) = (row as usize, col as usize);
            if row < 0 || col < 0 || r >= h.rows || c >= h.cols {
                Err(PyIndexError::new_err("index out of range"))
            } else {
                Ok(h.bin(r, c))
            }
        })?
    }
    fn get_hist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.with(py, |h| out_2d_f(py, h.rows, h.cols, |i, j| h.array[j * h.rows + i]))
    }
    fn get_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i32>>> {
        self.with(py, |h| out_2d_f(py, h.rows, h.cols, |i, j| h.counts[j * h.rows + i]))
    }
    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        self.with_mut(py, |h| h.reset())
    }
}

#[pyclass(name = "Hit", module = "pyroomacoustics.libroom")]
pub struct PyHit {
    distance: f32,
    transmitted: Vec<f32>,
}

#[pymethods]
impl PyHit {
    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        if args.len() == 1 {
            let a = args.get_item(0)?;
            if is_int(&a) {
                let n: i64 = a.extract()?;
                return Ok(PyHit { distance: 0.0, transmitted: vec![1.0; n.max(0) as usize] });
            }
        } else if args.len() == 2 {
            let d: f32 = args.get_item(0)?.extract()?;
            let t = vec_dyn(&args.get_item(1)?)?;
            return Ok(PyHit { distance: d, transmitted: t });
        }
        Err(PyTypeError::new_err("__init__(): incompatible constructor arguments"))
    }
    #[getter]
    fn transmitted<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        readonly(out_1d(py, &self.transmitted))
    }
    #[getter]
    fn distance(&self) -> f32 {
        self.distance
    }
}

/// `Wall.Isect`: an int-valued enum printing like the pybind11 enum.
fn isect_enum(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    static ISECT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    ISECT
        .get_or_try_init(py, || {
            let ns = PyDict::new(py);
            py.run(
                c"import enum\nclass Isect(int, enum.Enum):\n    NONE = -1\n    VALID = 0\n    ENDPT = 1\n    BNDRY = 2\nIsect.__module__ = 'pyroomacoustics.libroom'\nIsect.__qualname__ = 'Wall.Isect'\n",
                Some(&ns),
                Some(&ns),
            )?;
            Ok(ns.get_item("Isect")?.unwrap().unbind())
        })
        .map(|o| o.bind(py).clone())
}

fn isect_member(py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
    Ok(isect_enum(py)?.getattr(name)?.unbind())
}

// ---------------------------------------------------------------------------
// Dimension dependent classes (Wall/Wall2D, Microphone/Microphone2D, Room/Room2D)
// ---------------------------------------------------------------------------

macro_rules! dim_classes {
    ($D:literal, $Wall:ident, $wname:literal, $Mic:ident, $mname:literal, $Room:ident, $rname:literal, $Owner:ident) => {
        #[pyclass(name = $wname, module = "pyroomacoustics.libroom")]
        #[derive(Clone)]
        pub struct $Wall {
            pub inner: WallData<$D>,
        }

        #[allow(non_snake_case)]
        #[pymethods]
        impl $Wall {
            #[new]
            #[pyo3(signature = (corners, absorption=None, scattering=None, name=String::new()))]
            fn new(
                corners: &Bound<'_, PyAny>,
                absorption: Option<&Bound<'_, PyAny>>,
                scattering: Option<&Bound<'_, PyAny>>,
                name: String,
            ) -> PyResult<Self> {
                let c = mat_cols::<$D>(corners)?;
                let a = match absorption {
                    Some(a) => vec_dyn(a)?,
                    None => vec![0.0],
                };
                let s = match scattering {
                    Some(s) => vec_dyn(s)?,
                    None => vec![0.0],
                };
                let inner = WallData::<$D>::new(c, a, s, name).map_err(|e| PyRuntimeError::new_err(e.0))?;
                Ok($Wall { inner })
            }

            #[classattr]
            fn dim() -> i32 {
                $D
            }
            #[classattr]
            fn Isect(py: Python<'_>) -> PyResult<Py<PyAny>> {
                Ok(isect_enum(py)?.unbind())
            }
            #[classattr]
            fn NONE(py: Python<'_>) -> PyResult<Py<PyAny>> {
                isect_member(py, "NONE")
            }
            #[classattr]
            fn VALID(py: Python<'_>) -> PyResult<Py<PyAny>> {
                isect_member(py, "VALID")
            }
            #[classattr]
            fn ENDPT(py: Python<'_>) -> PyResult<Py<PyAny>> {
                isect_member(py, "ENDPT")
            }
            #[classattr]
            fn BNDRY(py: Python<'_>) -> PyResult<Py<PyAny>> {
                isect_member(py, "BNDRY")
            }

            fn area(&self) -> f32 {
                self.inner.area()
            }

            fn intersection(
                &self,
                p1: &Bound<'_, PyAny>,
                p2: &Bound<'_, PyAny>,
                intersection: &Bound<'_, PyAny>,
            ) -> PyResult<i32> {
                let a = vec_fixed::<$D>(p1)?;
                let b = vec_fixed::<$D>(p2)?;
                let mut out = read_out::<$D>(intersection)?;
                let r = self.inner.intersection(&a, &b, &mut out);
                write_out::<$D>(intersection, &out)?;
                Ok(r)
            }

            fn intersects(&self, p1: &Bound<'_, PyAny>, p2: &Bound<'_, PyAny>) -> PyResult<i32> {
                let a = vec_fixed::<$D>(p1)?;
                let b = vec_fixed::<$D>(p2)?;
                Ok(self.inner.intersects(&a, &b))
            }

            fn side(&self, p: &Bound<'_, PyAny>) -> PyResult<i32> {
                Ok(self.inner.side(&vec_fixed::<$D>(p)?))
            }

            fn reflect(&self, p: &Bound<'_, PyAny>, p_reflected: &Bound<'_, PyAny>) -> PyResult<i32> {
                let a = vec_fixed::<$D>(p)?;
                let mut out = read_out::<$D>(p_reflected)?;
                let r = self.inner.reflect(&a, &mut out);
                write_out::<$D>(p_reflected, &out)?;
                Ok(r)
            }

            #[pyo3(signature = (*args))]
            fn normal_reflect<'py>(
                &self,
                py: Python<'py>,
                args: &Bound<'py, PyTuple>,
            ) -> PyResult<Bound<'py, PyArray1<f32>>> {
                match args.len() {
                    3 => {
                        let s = vec_fixed::<$D>(&args.get_item(0)?)?;
                        let h = vec_fixed::<$D>(&args.get_item(1)?)?;
                        let l: f32 = args.get_item(2)?.extract()?;
                        Ok(out_1d(py, &self.inner.normal_reflect(&s, &h, l)))
                    }
                    1 => {
                        let i = vec_fixed::<$D>(&args.get_item(0)?)?;
                        Ok(out_1d(py, &self.inner.normal_reflect_dir(&i)))
                    }
                    _ => Err(PyTypeError::new_err("normal_reflect(): incompatible function arguments")),
                }
            }

            fn sample_lambertian_reflection<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
                let v = rng::with_engine(|r| self.inner.sample_lambertian_reflection(r));
                out_1d(py, &v)
            }

            fn same_as(&self, that: PyRef<'_, $Wall>) -> bool {
                self.inner.same_as(&that.inner)
            }

            #[getter]
            fn get_absorption<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                readonly(out_1d(py, &self.inner.absorption))
            }
            #[setter]
            fn set_absorption(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
                self.inner.absorption = vec_dyn(v)?;
                Ok(())
            }
            #[getter]
            fn get_scatter<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                readonly(out_1d(py, &self.inner.scatter))
            }
            #[setter]
            fn set_scatter(&mut self, v: &Bound<'_, PyAny>) -> PyResult<()> {
                self.inner.scatter = vec_dyn(v)?;
                Ok(())
            }
            #[getter]
            fn get_name(&self) -> String {
                self.inner.name.clone()
            }
            #[setter]
            fn set_name(&mut self, v: String) {
                self.inner.name = v;
            }
            #[getter]
            fn corners<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let c = &self.inner.corners;
                readonly(out_2d_f(py, $D, c.len(), |i, j| c[j][i]))
            }
            #[getter]
            fn origin<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                readonly(out_1d(py, &self.inner.origin))
            }
            #[getter]
            fn normal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                readonly(out_1d(py, &self.inner.normal))
            }
            #[getter]
            fn basis<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let b = &self.inner.basis;
                readonly(out_2d_f(py, $D, 2, |i, j| b[j][i]))
            }
            #[getter]
            fn flat_corners<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let f = &self.inner.flat_corners;
                readonly(out_2d_f(py, 2, f.len(), |i, j| f[j][i]))
            }
        }

        #[pyclass(name = $mname, module = "pyroomacoustics.libroom")]
        #[derive(Clone)]
        pub struct $Mic {
            pub inner: MicData<$D>,
        }

        #[pymethods]
        impl $Mic {
            #[new]
            fn new(loc: &Bound<'_, PyAny>, n_bands: i32, hist_res: f32, max_dist_init: f32) -> PyResult<Self> {
                let l = vec_fixed::<$D>(loc)?;
                Ok($Mic { inner: MicData::<$D>::new(l, n_bands.max(0) as usize, hist_res, max_dist_init) })
            }

            fn set_directions(&mut self, directions: &Bound<'_, PyAny>) -> PyResult<()> {
                let (r, c, data) = mat_dyn(directions)?;
                if c != $D {
                    return Err(PyRuntimeError::new_err(
                        "Error: 'dimensionality' must match column count in data matrix",
                    ));
                }
                let dirs: Vec<[f32; $D]> = (0..r)
                    .map(|i| {
                        let mut v = [0.0f32; $D];
                        v.copy_from_slice(&data[i * c..(i + 1) * c]);
                        v
                    })
                    .collect();
                self.inner.set_directions(dirs);
                Ok(())
            }

            fn make_omnidirectional(&mut self) {
                self.inner.make_omni();
            }

            fn log_histogram(
                &mut self,
                distance: f32,
                energy: &Bound<'_, PyAny>,
                origin: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let e = vec_dyn(energy)?;
                let o = vec_fixed::<$D>(origin)?;
                self.inner.log_histogram(distance, &e, &o);
                Ok(())
            }

            #[getter]
            fn loc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                readonly(out_1d(py, &self.inner.loc))
            }

            #[getter]
            fn histograms<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyList>> {
                let py = slf.py();
                let n = slf.borrow().inner.histograms.len();
                let items: PyResult<Vec<Bound<'py, PyHistogram2D>>> = (0..n)
                    .map(|i| Bound::new(py, PyHistogram2D { owner: HistOwner::$Owner(slf.clone().unbind(), i) }))
                    .collect();
                PyList::new(py, items?)
            }
        }

        #[pyclass(name = $rname, module = "pyroomacoustics.libroom")]
        pub struct $Room {
            geo: Geometry<$D>,
            walls: Vec<Py<$Wall>>,
            mics: Vec<Py<$Mic>>,
            params: Params,
            ism: IsmResults<$D>,
        }

        impl $Room {
            fn sync_walls(&mut self, py: Python<'_>) {
                self.geo.walls = self.walls.iter().map(|w| w.borrow(py).inner.clone()).collect();
            }

            fn mic_locs(&self, py: Python<'_>) -> Vec<[f32; $D]> {
                self.mics.iter().map(|m| m.borrow(py).inner.loc).collect()
            }

            fn build(
                py: Python<'_>,
                mut geo: Geometry<$D>,
                mics: Vec<MicData<$D>>,
                params: Params,
            ) -> PyResult<Self> {
                geo.init().map_err(|e| PyRuntimeError::new_err(e.0))?;
                let walls = geo
                    .walls
                    .iter()
                    .map(|w| Py::new(py, $Wall { inner: w.clone() }))
                    .collect::<PyResult<Vec<_>>>()?;
                let mics = mics
                    .into_iter()
                    .map(|m| Py::new(py, $Mic { inner: m }))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok($Room { geo, walls, mics, params, ism: IsmResults::default() })
            }

            /// Trace a batch of rays and log the result into the microphones.
            fn trace(
                &mut self,
                py: Python<'_>,
                dirs: Vec<[f32; $D]>,
                energies: tracer::Energies,
                source_pos: [f32; $D],
            ) -> PyResult<()> {
                self.sync_walls(py);
                let mut mic_refs: Vec<PyRefMut<'_, $Mic>> = self.mics.iter().map(|m| m.borrow_mut(py)).collect();
                let geoms: Vec<MicData<$D>> = mic_refs
                    .iter()
                    .map(|m| {
                        let mut g = m.inner.clone();
                        g.histograms = vec![];
                        g
                    })
                    .collect();
                let mut hists: Vec<&mut Vec<HistData>> =
                    mic_refs.iter_mut().map(|m| &mut m.inner.histograms).collect();
                let (geo, params) = (&self.geo, &self.params);
                pool::install(|| tracer::trace_rays(geo, params, &geoms, &mut hists, &dirs, &energies, &source_pos));
                Ok(())
            }
        }

        #[pymethods]
        impl $Room {
            #[new]
            #[pyo3(signature = (*args))]
            fn new(py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<Self> {
                let n = args.len();
                let mic_list = |obj: &Bound<'_, PyAny>| -> PyResult<Vec<MicData<$D>>> {
                    let l = obj.cast::<PyList>().map_err(|_| type_err())?;
                    l.iter().map(|m| Ok(m.cast::<$Mic>().map_err(|_| type_err())?.borrow().inner.clone())).collect()
                };
                let params_from = |off: usize| -> PyResult<Params> {
                    Ok(Params::new(
                        args.get_item(off)?.extract::<f32>()?,
                        args.get_item(off + 1)?.extract::<i32>()?,
                        args.get_item(off + 2)?.extract::<f32>()?,
                        args.get_item(off + 3)?.extract::<f32>()?,
                        args.get_item(off + 4)?.extract::<f32>()?,
                        args.get_item(off + 5)?.extract::<f32>()?,
                        args.get_item(off + 6)?.extract::<bool>()?,
                    ))
                };
                if n == 10 {
                    let wl = args.get_item(0)?;
                    let wl = wl.cast::<PyList>().map_err(|_| type_err())?;
                    let walls: Vec<WallData<$D>> = wl
                        .iter()
                        .map(|w| Ok(w.cast::<$Wall>().map_err(|_| type_err())?.borrow().inner.clone()))
                        .collect::<PyResult<_>>()?;
                    let obstructing: Vec<i32> = args.get_item(1)?.extract()?;
                    let mics = mic_list(&args.get_item(2)?)?;
                    let params = params_from(3)?;
                    let geo = Geometry::<$D> {
                        walls,
                        obstructing_walls: obstructing,
                        is_shoebox: false,
                        shoebox_size: [0.0; $D],
                        shoebox_absorption: vec![],
                        shoebox_scattering: vec![],
                        n_bands: 0,
                        max_dist: 0.0,
                    };
                    $Room::build(py, geo, mics, params)
                } else if n == 11 {
                    let size = vec_fixed::<$D>(&args.get_item(0)?)?;
                    let (ar, ac, ad) = mat_dyn(&args.get_item(1)?)?;
                    let (sr, sc, sd) = mat_dyn(&args.get_item(2)?)?;
                    if ac != 2 * $D || sc != 2 * $D {
                        return Err(type_err());
                    }
                    let mics = mic_list(&args.get_item(3)?)?;
                    let params = params_from(4)?;
                    if ar != sr {
                        return Err(PyRuntimeError::new_err(
                            "Error: The same number of absorption and scattering coefficients are required",
                        ));
                    }
                    let absm: Vec<Vec<f32>> = (0..ar).map(|i| ad[i * ac..(i + 1) * ac].to_vec()).collect();
                    let scam: Vec<Vec<f32>> = (0..sr).map(|i| sd[i * sc..(i + 1) * sc].to_vec()).collect();
                    let walls = room::make_shoebox_walls::<$D>(&size, &absm, &scam)
                        .map_err(|e| PyRuntimeError::new_err(e.0))?;
                    let geo = Geometry::<$D> {
                        walls,
                        obstructing_walls: vec![],
                        is_shoebox: true,
                        shoebox_size: size,
                        shoebox_absorption: absm,
                        shoebox_scattering: scam,
                        n_bands: 0,
                        max_dist: 0.0,
                    };
                    $Room::build(py, geo, mics, params)
                } else {
                    Err(PyTypeError::new_err("__init__(): incompatible constructor arguments"))
                }
            }

            fn set_params(
                &mut self,
                sound_speed: f32,
                ism_order: i32,
                energy_thres: f32,
                time_thres: f32,
                mic_radius: f32,
                mic_hist_res: f32,
                is_hybrid_sim: bool,
            ) {
                self.params = Params::new(
                    sound_speed,
                    ism_order,
                    energy_thres,
                    time_thres,
                    mic_radius,
                    mic_hist_res,
                    is_hybrid_sim,
                );
            }

            fn add_mic(&mut self, py: Python<'_>, loc: &Bound<'_, PyAny>) -> PyResult<()> {
                let l = vec_fixed::<$D>(loc)?;
                let p = &self.params;
                let m = MicData::<$D>::new(
                    l,
                    self.geo.n_bands,
                    p.mic_hist_res * p.sound_speed,
                    p.time_thres * p.sound_speed,
                );
                self.mics.push(Py::new(py, $Mic { inner: m })?);
                Ok(())
            }

            fn reset_mics(&mut self, py: Python<'_>) {
                for m in &self.mics {
                    m.borrow_mut(py).inner.reset();
                }
            }

            fn broadcast_bands_to(&mut self, py: Python<'_>, new_n_bands: i32) -> PyResult<()> {
                let nb = new_n_bands.max(0) as usize;
                self.geo.n_bands = nb;
                for w in &self.walls {
                    w.borrow_mut(py).inner.broadcast_bands_to(nb).map_err(|e| PyRuntimeError::new_err(e.0))?;
                }
                for m in &self.mics {
                    m.borrow_mut(py).inner.reset_bands(nb);
                }
                Ok(())
            }

            fn image_source_model(&mut self, py: Python<'_>, source_location: &Bound<'_, PyAny>) -> PyResult<i32> {
                let s = vec_fixed::<$D>(source_location)?;
                self.sync_walls(py);
                let mics = self.mic_locs(py);
                let geo = &self.geo;
                let params = &self.params;
                let ism = &mut self.ism;
                Ok(py.detach(|| pool::install(|| geo.image_source_model(params, &s, &mics, ism))))
            }

            fn get_wall(&self, py: Python<'_>, w: i64) -> PyResult<$Wall> {
                if w < 0 || w as usize >= self.walls.len() {
                    return Err(PyIndexError::new_err("wall index out of range"));
                }
                Ok(self.walls[w as usize].borrow(py).clone())
            }

            fn get_max_distance(&mut self, py: Python<'_>) -> f32 {
                self.sync_walls(py);
                self.geo.get_max_distance()
            }

            fn next_wall_hit<'py>(
                &mut self,
                py: Python<'py>,
                start: &Bound<'py, PyAny>,
                end: &Bound<'py, PyAny>,
                scattered_ray: bool,
            ) -> PyResult<(Bound<'py, PyArray1<f32>>, i32, f32)> {
                let s = vec_fixed::<$D>(start)?;
                let e = vec_fixed::<$D>(end)?;
                self.sync_walls(py);
                let (r, i, d) = self.geo.next_wall_hit(&s, &e, scattered_ray);
                Ok((out_1d(py, &r), i, d))
            }

            fn scat_ray(
                &mut self,
                py: Python<'_>,
                transmitted: &Bound<'_, PyAny>,
                wall: PyRef<'_, $Wall>,
                prev_last_hit: &Bound<'_, PyAny>,
                hit_point: &Bound<'_, PyAny>,
                travel_dist: f32,
            ) -> PyResult<bool> {
                let t = vec_dyn(transmitted)?;
                let plh = vec_fixed::<$D>(prev_last_hit)?;
                let hp = vec_fixed::<$D>(hit_point)?;
                let w = wall.inner.clone();
                drop(wall);
                self.sync_walls(py);
                let mut mic_refs: Vec<PyRefMut<'_, $Mic>> = self.mics.iter().map(|m| m.borrow_mut(py)).collect();
                let geoms: Vec<MicData<$D>> = mic_refs.iter().map(|m| m.inner.clone()).collect();
                let mut log = room::EventLog::default();
                let mut buf = vec![];
                let r = self.geo.scat_ray(&self.params, &geoms, &mut log, &t, &w, &plh, &hp, travel_dist, &mut buf);
                let nb = if log.events.is_empty() { 0 } else { log.energies.len() / log.events.len() };
                for (i, e) in log.events.iter().enumerate() {
                    mic_refs[e.mic as usize].inner.histograms[e.dir as usize]
                        .log_col(e.col as usize, &log.energies[i * nb..(i + 1) * nb]);
                }
                Ok(r)
            }

            #[pyo3(signature = (*args))]
            fn simul_ray(&mut self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<()> {
                let (dir, src, e0) = match args.len() {
                    4 => {
                        let phi: f32 = args.get_item(0)?.extract()?;
                        let theta: f32 = args.get_item(1)?.extract()?;
                        let src = vec_fixed::<$D>(&args.get_item(2)?)?;
                        let e0 = vec_dyn(&args.get_item(3)?)?;
                        (tracer::angles_to_dir::<$D>(phi, theta), src, e0)
                    }
                    3 => {
                        let d = vec_fixed::<$D>(&args.get_item(0)?)?;
                        let src = vec_fixed::<$D>(&args.get_item(1)?)?;
                        let e0 = vec_dyn(&args.get_item(2)?)?;
                        (d, src, e0)
                    }
                    _ => return Err(PyTypeError::new_err("simul_ray(): incompatible function arguments")),
                };
                self.trace(py, vec![dir], tracer::Energies::PerRay(vec![e0]), src)
            }

            #[pyo3(signature = (*args))]
            fn ray_tracing(&mut self, py: Python<'_>, args: &Bound<'_, PyTuple>) -> PyResult<()> {
                let nb = self.geo.n_bands;
                match args.len() {
                    2 => {
                        let a0 = args.get_item(0)?;
                        let src = vec_fixed::<$D>(&args.get_item(1)?)?;
                        if is_int(&a0) {
                            let n_rays: u64 = a0.extract()?;
                            let (dirs, ev) = tracer::fibonacci_dirs::<$D>(n_rays as usize);
                            self.trace(py, dirs, tracer::Energies::Uniform(vec![ev; nb]), src)
                        } else {
                            // angles matrix (D-1, n)
                            let (shape, data) = f32s(&a0)?;
                            let (rows, cols) = match shape.len() {
                                1 if $D == 2 => (1, shape[0]),
                                2 => (shape[0], shape[1]),
                                _ => return Err(type_err()),
                            };
                            if rows != $D - 1 {
                                return Err(type_err());
                            }
                            let ev = wall::ENERGY_0 / (cols as f32);
                            let dirs: Vec<[f32; $D]> = (0..cols)
                                .map(|k| {
                                    let phi = data[k];
                                    let theta = if $D == 3 { data[cols + k] } else { wall::HALF_PI };
                                    tracer::angles_to_dir::<$D>(phi, theta)
                                })
                                .collect();
                            self.trace(py, dirs, tracer::Energies::Uniform(vec![ev; nb]), src)
                        }
                    }
                    3 => {
                        let a0 = args.get_item(0)?;
                        let a1 = args.get_item(1)?;
                        let src = vec_fixed::<$D>(&args.get_item(2)?)?;
                        if is_int(&a0) && is_int(&a1) {
                            let nb_phis: u64 = a0.extract()?;
                            let nb_thetas: u64 = a1.extract()?;
                            let (dirs, ev) = tracer::grid_dirs::<$D>(nb_phis as usize, nb_thetas as usize);
                            self.trace(py, dirs, tracer::Energies::Uniform(vec![ev; nb]), src)
                        } else {
                            let vecs = mat_rows::<$D>(&a0)?;
                            let (er, ec, ed) = mat_dyn(&a1)?;
                            let e0 = wall::ENERGY_0 / (vecs.len() as f32);
                            if vecs.len() != er {
                                return Err(PyRuntimeError::new_err(
                                    "Error: The same number of rays and energies is expected.",
                                ));
                            }
                            let energies: Vec<Vec<f32>> =
                                (0..er).map(|r| ed[r * ec..(r + 1) * ec].iter().map(|x| e0 * x).collect()).collect();
                            self.trace(py, vecs, tracer::Energies::PerRay(energies), src)
                        }
                    }
                    _ => Err(PyTypeError::new_err("ray_tracing(): incompatible function arguments")),
                }
            }

            fn contains(&mut self, py: Python<'_>, point: &Bound<'_, PyAny>) -> PyResult<bool> {
                let p = vec_fixed::<$D>(point)?;
                self.sync_walls(py);
                Ok(rng::with_engine(|r| self.geo.contains(&p, r)))
            }

            #[getter]
            fn get_is_hybrid_sim(&self) -> bool {
                self.params.is_hybrid_sim
            }
            #[setter]
            fn set_is_hybrid_sim(&mut self, v: bool) {
                self.params.is_hybrid_sim = v;
            }

            #[classattr]
            fn dim() -> i32 {
                $D
            }

            #[getter]
            fn walls<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
                PyList::new(py, self.walls.iter().map(|w| w.clone_ref(py)))
            }
            #[getter]
            fn microphones<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
                PyList::new(py, self.mics.iter().map(|m| m.clone_ref(py)))
            }
            #[getter]
            fn obstructing_walls(&self) -> Vec<i32> {
                self.geo.obstructing_walls.clone()
            }
            #[getter]
            fn max_dist(&self) -> f32 {
                self.geo.max_dist
            }
            #[getter]
            fn sources<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let s = &self.ism.sources;
                readonly(out_2d_f(py, $D, s.len(), |i, j| s[j][i]))
            }
            #[getter]
            fn orders<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                readonly(out_1d(py, &self.ism.orders))
            }
            #[getter]
            fn gen_walls<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                readonly(out_1d(py, &self.ism.gen_walls))
            }
            #[getter]
            fn orders_xyz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let s = &self.ism.orders_xyz;
                readonly(out_2d_f(py, $D, s.len(), |i, j| s[j][i]))
            }
            #[getter]
            fn source_directions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let s = &self.ism.source_directions;
                readonly(out_2d_f(py, $D, s.len(), |i, j| s[j][i]))
            }
            #[getter]
            fn attenuations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let a = &self.ism.attenuations;
                let n = self.ism.n_sources;
                let nb = if n == 0 { 0 } else { self.ism.n_bands };
                readonly(out_2d_f(py, nb, n, |i, j| a[j * nb + i]))
            }
            #[getter]
            fn visible_mics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                let v = &self.ism.visible_mics;
                let n = self.ism.n_sources;
                let nm = if n == 0 { 0 } else { self.ism.n_mics };
                readonly(out_2d_f(py, nm, n, |i, j| v[j * nm + i]))
            }
        }
    };
}

dim_classes!(3, PyWall3, "Wall", PyMic3, "Microphone", PyRoom3, "Room", Mic3);
dim_classes!(2, PyWall2, "Wall2D", PyMic2, "Microphone2D", PyRoom2, "Room2D", Mic2);

// ---------------------------------------------------------------------------
// free functions
// ---------------------------------------------------------------------------

fn v2(o: &Bound<'_, PyAny>) -> PyResult<[f32; 2]> {
    vec_fixed::<2>(o)
}
fn v3(o: &Bound<'_, PyAny>) -> PyResult<[f32; 3]> {
    vec_fixed::<3>(o)
}

#[pyfunction]
fn set_rng_seed(seed: u64) {
    rng::set_seed(seed);
}

/// Use one random generator per ray so that ray tracing with scattering runs in
/// parallel. Results are statistically equivalent to the original but not
/// bit-identical. Default: True; `constants.set("exact_mode", True)` turns it off.
#[pyfunction]
fn set_parallel_rng(state: bool) {
    tracer::PARALLEL_RNG.store(state, std::sync::atomic::Ordering::Relaxed);
}

/// Maximum number of worker threads used by the compiled kernels.
#[pyfunction]
fn set_num_threads(n: usize) {
    pool::set_num_threads(n);
}

#[pyfunction]
fn get_num_threads() -> usize {
    pool::num_threads()
}

#[pyfunction]
fn get_parallel_rng() -> bool {
    tracer::PARALLEL_RNG.load(std::sync::atomic::Ordering::Relaxed)
}

#[pyfunction]
fn set_eps(eps: f32) {
    geom::set_eps(eps);
}

#[pyfunction]
fn get_eps() -> f32 {
    geom::eps()
}

#[pyfunction]
fn ccw3p(p1: &Bound<'_, PyAny>, p2: &Bound<'_, PyAny>, p3: &Bound<'_, PyAny>) -> PyResult<i32> {
    Ok(geom::ccw3p(&v2(p1)?, &v2(p2)?, &v2(p3)?))
}

#[pyfunction]
fn check_intersection_2d_segments(
    a1: &Bound<'_, PyAny>,
    a2: &Bound<'_, PyAny>,
    b1: &Bound<'_, PyAny>,
    b2: &Bound<'_, PyAny>,
) -> PyResult<i32> {
    Ok(geom::check_intersection_2d_segments(&v2(a1)?, &v2(a2)?, &v2(b1)?, &v2(b2)?))
}

#[pyfunction]
fn intersection_2d_segments(
    a1: &Bound<'_, PyAny>,
    a2: &Bound<'_, PyAny>,
    b1: &Bound<'_, PyAny>,
    b2: &Bound<'_, PyAny>,
    intersection: &Bound<'_, PyAny>,
) -> PyResult<i32> {
    let (a1, a2, b1, b2) = (v2(a1)?, v2(a2)?, v2(b1)?, v2(b2)?);
    let mut out = read_out::<2>(intersection)?;
    let r = geom::intersection_2d_segments(&a1, &a2, &b1, &b2, &mut out);
    write_out::<2>(intersection, &out)?;
    Ok(r)
}

#[pyfunction]
fn intersection_3d_segment_plane(
    a1: &Bound<'_, PyAny>,
    a2: &Bound<'_, PyAny>,
    p: &Bound<'_, PyAny>,
    normal: &Bound<'_, PyAny>,
    intersection: &Bound<'_, PyAny>,
) -> PyResult<i32> {
    let (a1, a2, p, n) = (v3(a1)?, v3(a2)?, v3(p)?, v3(normal)?);
    let mut out = read_out::<3>(intersection)?;
    let r = geom::intersection_3d_segment_plane(&a1, &a2, &p, &n, &mut out);
    write_out::<3>(intersection, &out)?;
    Ok(r)
}

#[pyfunction]
fn cross<'py>(py: Python<'py>, v1: &Bound<'py, PyAny>, v2: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    Ok(out_1d(py, &eig::cross(&v3(v1)?, &v3(v2)?)))
}

#[pyfunction]
fn is_inside_2d_polygon(p: &Bound<'_, PyAny>, corners: &Bound<'_, PyAny>) -> PyResult<i32> {
    let c = mat_cols::<2>(corners)?;
    Ok(geom::is_inside_2d_polygon(&v2(p)?, &c))
}

#[pyfunction]
fn area_2d_polygon(corners: &Bound<'_, PyAny>) -> PyResult<f32> {
    let c = mat_cols::<2>(corners)?;
    Ok(geom::area_2d_polygon(&c))
}

#[pyfunction]
fn cos_angle_between(v1: &Bound<'_, PyAny>, v2: &Bound<'_, PyAny>) -> PyResult<f32> {
    Ok(geom::cos_angle_between(&vec_dyn(v1)?, &vec_dyn(v2)?))
}

#[pyfunction]
fn dist_line_point(start: &Bound<'_, PyAny>, end: &Bound<'_, PyAny>, point: &Bound<'_, PyAny>) -> PyResult<f32> {
    Ok(geom::dist_line_point(&vec_dyn(start)?, &vec_dyn(end)?, &vec_dyn(point)?))
}

/// pybind11 `py::buffer` arguments only accept objects exposing the buffer protocol.
fn require_buffer(objs: &[&Bound<'_, PyAny>]) -> PyResult<()> {
    for o in objs {
        if pyo3::types::PyMemoryView::from(o).is_err() {
            return Err(type_err());
        }
    }
    Ok(())
}

fn dtype_is(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    let py = obj.py();
    let np = numpy(py)?;
    if !obj.is_instance(&np.getattr("ndarray")?)? {
        return Ok(false);
    }
    let dt = obj.getattr("dtype")?;
    dt.eq(np.getattr(name)?)
}

/// Run `f` on the data of `arr` (dtype T). If `arr` is not a C-contiguous
/// array of T, the original works on a temporary copy (forcecast) and the
/// result is discarded: we mimic this.
fn with_inplace<T: Element + Copy, R>(
    arr: &Bound<'_, PyAny>,
    dtype: &str,
    f: impl FnOnce(&mut [T], &[usize]) -> R,
) -> PyResult<R> {
    if let Ok(a) = arr.cast::<PyArrayDyn<T>>() {
        if a.is_c_contiguous() {
            let shape = a.shape().to_vec();
            let mut rw = a.try_readwrite().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let s = rw.as_slice_mut().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            return Ok(f(s, &shape));
        }
    }
    let (shape, mut data) = to_array::<T>(arr, dtype)?;
    Ok(f(&mut data, &shape))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rir_builder(
    py: Python<'_>,
    rir: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
    alpha: &Bound<'_, PyAny>,
    fs: i32,
    fdl: usize,
    lut_gran: usize,
    num_threads: usize,
) -> PyResult<()> {
    require_buffer(&[rir, time, alpha])?;
    if dtype_is(rir, "float32")? {
        let (_, t) = to_array::<f32>(time, "float32")?;
        let (_, a) = to_array::<f32>(alpha, "float32")?;
        with_inplace::<f32, _>(rir, "float32", |r, _| {
            py.detach(|| pool::install(|| rir::rir_builder(r, &t, &a, fs, fdl, lut_gran, num_threads)))
        })?
        .map_err(|e| PyRuntimeError::new_err(e.0))
    } else if dtype_is(rir, "float64")? {
        let (_, t) = to_array::<f64>(time, "float64")?;
        let (_, a) = to_array::<f64>(alpha, "float64")?;
        with_inplace::<f64, _>(rir, "float64", |r, _| {
            py.detach(|| pool::install(|| rir::rir_builder(r, &t, &a, fs, fdl, lut_gran, num_threads)))
        })?
        .map_err(|e| PyRuntimeError::new_err(e.0))
    } else {
        Ok(())
    }
}

#[pyfunction]
fn delay_sum(
    py: Python<'_>,
    irs: &Bound<'_, PyAny>,
    delays: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    num_threads: usize,
) -> PyResult<()> {
    require_buffer(&[irs, delays, output])?;
    let (_, d) = to_array::<i32>(delays, "int32")?;
    macro_rules! go {
        ($T:ty, $name:literal) => {{
            let (shape, data) = to_array::<$T>(irs, $name)?;
            if shape.len() != 2 {
                return Err(type_err());
            }
            with_inplace::<$T, _>(output, $name, |o, _| {
                py.detach(|| pool::install(|| rir::delay_sum(&data, shape[0], shape[1], &d, o, num_threads)))
            })?
            .map_err(|e| PyRuntimeError::new_err(e.0))
        }};
    }
    if dtype_is(irs, "float32")? {
        go!(f32, "float32")
    } else if dtype_is(irs, "float64")? {
        go!(f64, "float64")
    } else {
        Ok(())
    }
}

#[pyfunction]
fn fractional_delay(
    py: Python<'_>,
    out: &Bound<'_, PyAny>,
    time: &Bound<'_, PyAny>,
    lut_gran: usize,
    num_threads: usize,
) -> PyResult<()> {
    let _ = num_threads;
    require_buffer(&[out, time])?;
    macro_rules! go {
        ($T:ty, $name:literal) => {{
            let (_, t) = to_array::<$T>(time, $name)?;
            with_inplace::<$T, _>(out, $name, |o, shape| {
                if shape.len() != 2 {
                    return Err(rir::RirError("output array needs to be 2D".into()));
                }
                py.detach(|| pool::install(|| rir::fractional_delay(o, shape[0], shape[1], &t, lut_gran)))
            })?
            .map_err(|e| PyRuntimeError::new_err(e.0))
        }};
    }
    if dtype_is(out, "float32")? {
        go!(f32, "float32")
    } else if dtype_is(out, "float64")? {
        go!(f64, "float64")
    } else {
        Ok(())
    }
}

#[pymodule]
fn libroom(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Libroom room simulation extension plugin (Rust implementation)")?;
    m.add_function(wrap_pyfunction!(set_rng_seed, m)?)?;
    m.add_class::<PyRoom3>()?;
    m.add_class::<PyRoom2>()?;
    m.add_class::<PyWall3>()?;
    m.add_class::<PyWall2>()?;
    m.add_class::<PyMic3>()?;
    m.add_class::<PyMic2>()?;
    m.add_class::<PyHistogram2D>()?;
    m.add_class::<PyHit>()?;
    m.add("WALL_ISECT_NONE", geom::WALL_ISECT_NONE)?;
    m.add("WALL_ISECT_VALID", geom::WALL_ISECT_VALID)?;
    m.add("WALL_ISECT_VALID_ENDPT", geom::WALL_ISECT_VALID_ENDPT)?;
    m.add("WALL_ISECT_VALID_BNDRY", geom::WALL_ISECT_VALID_BNDRY)?;
    m.add_function(wrap_pyfunction!(set_eps, m)?)?;
    m.add_function(wrap_pyfunction!(set_parallel_rng, m)?)?;
    m.add_function(wrap_pyfunction!(get_parallel_rng, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_eps, m)?)?;
    m.add_function(wrap_pyfunction!(ccw3p, m)?)?;
    m.add_function(wrap_pyfunction!(check_intersection_2d_segments, m)?)?;
    m.add_function(wrap_pyfunction!(intersection_2d_segments, m)?)?;
    m.add_function(wrap_pyfunction!(intersection_3d_segment_plane, m)?)?;
    m.add_function(wrap_pyfunction!(cross, m)?)?;
    m.add_function(wrap_pyfunction!(is_inside_2d_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(area_2d_polygon, m)?)?;
    m.add_function(wrap_pyfunction!(cos_angle_between, m)?)?;
    m.add_function(wrap_pyfunction!(dist_line_point, m)?)?;
    m.add_function(wrap_pyfunction!(rir_builder, m)?)?;
    m.add_function(wrap_pyfunction!(delay_sum, m)?)?;
    m.add_function(wrap_pyfunction!(fractional_delay, m)?)?;
    rt_kernels::register(m)?;
    fast::register(m)?;
    let _ = rt_err::<&str>;
    Ok(())
}
