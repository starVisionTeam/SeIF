from pymeshfix import PyTMesh
mix = PyTMesh(False)
filename=''
mix.load_file(filename)
mix.fill_small_boundaries(refine=True)
filename = filename.replace("obj","_hole.obj")
mix.save_file(filename)
