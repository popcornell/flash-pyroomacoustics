import runpy, sys, pyroomacoustics as pra
pra.constants.set("exact_mode", True)
sys.argv = ["dump.py"] + sys.argv[1:]
runpy.run_path("dump.py", run_name="__main__")
