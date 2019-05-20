
#includes all relevant "modules"

#path to modules
path = "./core/"

include(string(path,"types.jl"))
include(string(path,"nonlins.jl"))
include(string(path,"helpers.jl"))
include(string(path,"dataimport.jl"))
include(string(path,"forwardprop.jl"))
include(string(path,"backprop_weightupdate.jl"))
include(string(path,"learning.jl"))
include(string(path,"training.jl"))
