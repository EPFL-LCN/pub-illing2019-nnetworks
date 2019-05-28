
using LambertW

"""
    struct NoNoise end
"""
struct NoNoise end
export NoNoise

"""
    mutable struct EscapeNoise
      β::Float64
      endofrefrperiod::Array{Float64, 1}
    end
"""
mutable struct RectLinEscapeNoise
  β::Float64
  endofrefrperiod::Array{Float64, 1}
end
export RectLinEscapeNoise
function RectLinEscapeNoise(array; β = 0.1)
  RectLinEscapeNoise(β, array)
end


@inline function _gettimetothreshold(v, input, threshold, tau, v_reset)
 (v < threshold) ? tau*log((input-v)/(input-threshold)) : 0
end
@inline function _getspiketimesampleparams(v, input, threshold, tau, beta, v_reset)
  a = beta*(input-threshold)
  b = beta*tau*(input-v)
  return a, b
end
@inline function getspiketimesample(v, input, threshold, t, tau, beta, v_reset, isinrefrperiod, t_refr_rest)
  threshold >= input && return typeof(v)(Inf)
  isinrefrperiod && (v = v_reset)
  #a, b = _getspiketimesampleparams(v, input, threshold, tau, beta, v_reset)
  y = -log(1-rand())
  a = beta*(input-threshold)
  if v < threshold
    tsample = tau*(1+lambertw(-exp(-1-y/(a*tau)))) + y/a +
        _gettimetothreshold(v, input, threshold, tau, v_reset)
  else
    b = beta*tau*(input-v)
    tsample = tau*lambertw(-b*exp(-(y+b)/(a*tau))/(a*tau)) + (b + y)/a
  end
  isinrefrperiod ? (tsample += t_refr_rest) : (tsample += t) #relative to absolute time
  return tsample
end
function _evaluateanalyticdistribution!(ISIdistr, v, input, threshold, tau, beta, v_reset, t_refr_rest, trise, i)
  if v < threshold
    if i <= trise+t_refr_rest
      push!(ISIdistr,0.)
    else
      push!(ISIdistr,beta*((input-threshold)-(input-v)*exp(-(i-t_refr_rest)/tau))*
      exp(-beta*(input-threshold)*(i-t_refr_rest-trise-tau)-beta*tau*(input-v)*(exp(-(i-t_refr_rest)/tau))))
    end
  else
    push!(ISIdistr,beta*(input-threshold-(input-v)*exp(-i/tau))*
    exp(-beta*i*(input-threshold)+beta*tau*(input-v)*(1-exp(-i/tau))))
  end
end
function getanalyticdistribution(v, input, threshold, t, tau, beta, v_reset, isinrefrperiod, t_refr_rest; t_lag = [])
  threshold >= input && error("input to small, no spike ever")
  isinrefrperiod && (v = v_reset)
  ISIdistr = []
  trise = _gettimetothreshold(v, input, threshold, tau, v_reset)
  if isempty(t_lag)
    for i in 0:0.01:100
      _evaluateanalyticdistribution!(ISIdistr, v, input, threshold, tau, beta, v_reset, t_refr_rest, trise, i)
      push!(t_lag,i)
    end
  else
    for i in t_lag
      _evaluateanalyticdistribution!(ISIdistr, v, input, threshold, tau, beta, v_reset, t_refr_rest, trise, i)
    end
  end
  t_lag .+= t
  return t_lag, ISIdistr, trise
end
export getanalyticdistribution
