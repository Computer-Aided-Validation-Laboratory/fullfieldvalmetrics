# Modified Area Validation Metric Calculation Notes:

## Symbols
Sn = Experiment
F = Simulation
Y = System Response Quantity (measure in experiment)

d = absolute value area between the model and experiment empirical CDFs

The true values lies between
$[F(Y)-d,F(Y)+d]$

## Scipy Stats Notes:
.quantiles = physical values corresponding to probabilities in measurement units
.probabilities = corresponding probabilities
plot(X,Y) = plot(quantiles,probabilities) = ecdf plot

## Notes on algorithm
- The line: p_f = 1/S is ambiguous, S without bold is specified as the mean simulation result in the nomenclature table. But based on p_sn it should be 1/num sims. Note sure which is correct here. Based on the line N > S this would be number of simulations vs number of experiments

## AM Implementation:

```python
def mavm(model_data,
         exp_data
         ) -> dict[str,Any]:
    """
    Calculates the Modified Area Validation Metric.
    Adapted from Whiting et al., 2023, "Assessment of Model Validation, Calibration, and Prediction Approaches in the Presence of Uncertainty", Journal of Verification, Validation and Uncertainty Quantification, Vol. 8.
    Downloaded from http://asmedigitalcollection.asme.org/verification/article-pdf/8/1/011001/6974199/vvuq_008_01_011001.pdf on 24 May 2024.
    """

    # find empirical cdf
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf

    F_ = model_cdf.quantiles
    Sn_ = exp_cdf.quantiles



    df = len(Sn_)-1
    t_alph = stats.t.ppf(0.95,df)

    Sn_conf = [Sn_ - t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_))),
               Sn_ + t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_)))]

    Sn_Y = exp_cdf.probabilities
    F_Y = model_cdf.probabilities

    P_F = 1/len(F_)
    P_Sn = 1/len(exp_cdf.quantiles)

    d_conf_plus = []
    d_conf_minus = []

    for k in [0,1]:

        ii = 0
        d_rem = 0

        d_plus = 0
        d_minus = 0

        Sn = Sn_conf[k]

        #If more experimental data points than model data points
        if len(Sn) > len(F_):

            for jj in range(0,len(F_)):
                if d_rem != 0:
                    d_ = (Sn[ii] - F_[jj]) * (P_Sn*(ii+1) - P_F*jj)
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_
                    ii += 1
                while (jj+1)*P_F > (ii+1)*P_Sn:
                    d_ = (Sn[ii] - F_[jj])*P_F
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_

                    ii += 1
                d_rem = (Sn[ii]-F_[jj])*(P_F*(jj+1) - P_Sn*ii)
                if d_rem > 0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem

        #If more model data points than experimental data points (more typical)
        elif len(Sn) <= len(F_):

            for jj in range(0,len(Sn)):

                if d_rem != 0:
                    d_ = (Sn[jj]-F_[ii])*(P_F*(ii+1) - P_Sn*jj)
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_
                    ii += 1

                while (ii+1)*P_F < (jj+1)*P_Sn:
                    d_ = (Sn[jj]-F_[ii])*P_F
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_

                    ii += 1

                d_rem = (Sn[jj]-F_[ii])*(P_Sn*(jj+1) - P_F*ii)
                if d_rem > 0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem

        d_conf_plus.append(np.abs(d_plus))
        d_conf_minus.append(np.abs(d_minus))

    d_plus = np.nanmax(d_conf_plus)
    d_minus = np.nanmax(d_conf_minus)


    output_dict = {"model_cdf":model_cdf,
                   "exp_cdf":exp_cdf,
                   "d+":d_plus,
                   "d-":d_minus,
                   "Sn_conf":Sn_conf,
                   "F_":F_,
                   "F_Y":F_Y,}

    return output_dict
```

