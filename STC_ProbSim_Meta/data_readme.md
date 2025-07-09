# Data Description
https://ukaeauk-my.sharepoint.com/:f:/g/personal/jonathan_horne-jones_ukaea_uk/Es5yfMPMxE1Mr7A6V3EXtCUBV2uYcx-QKRdAWGjc63CGgg?e=7fcWqt

I've now completed the DIC based position and orientation processing. This has turned out to be quite noisy (probably due to the variability in the point cloud registration to align the coil scan and DIC). Given the epistemic error (assuming to be +- 0.5*pixel spacing) is much smaller we are going to have to use a combined epistemic - aleatoric uncertainty propagation. To make things a little simpler, I have used the fact that the input uncertainty is always dominated by either epistemic or aleatoric uncertainty to avoid use of the Dempster-Shafer (overlapping interval analysis) method. For each output we now have an ensemble of CDFs (e.g. below). I have generated this with 400 aleatoric samples and 250 epistemic samples. The sampling is done in a nested loop - for each epistemic sample 400 aleatoric samples are drawn. This is reflected in the data, with each epistemic sample generating 400 rows of data.


All data from the probabilistic simulation is here: KC4. This includes:
https://ukaeauk-my.sharepoint.com/:f:/g/personal/jonathan_horne-jones_ukaea_uk/Es5yfMPMxE1Mr7A6V3EXtCUBV2uYcx-QKRdAWGjc63CGgg?e=7fcWqt
- The mesh for all reconstructed fields "Mesh.csv"
- The sample points for the simulation "SamplingDOE.csv" (you may not need this)
- The predicted results for the simulation "SamplingResults.csv"
	- These are grouped in 400 row chunks, each of which are the samples constructing a single CDF
	- Hopefully it will be reasonably simple to extract the upper and lower bounding CDF and compute the validation metric for each
- A numpy binary file for each variable predicted in field form
	- Rows = nodes (matching mesh file)
	- Columns = predicted fields


We want to compute validation metrics for the following:
- All working thermocouples
- Coil voltage
- All viable fields (not sure if we have reliable temperature field data)

Ideally these would include experimental epistemic error, which is recorded along with the aleatoric statistics here: STC2_1550A_SteadySummary.csv

## Sim Data Chunking

### Reduced data
For the 5000 field set there are 100 aleatoric samples and 50 epistemic samples. In both cases if you should chunk by number of aleatoric samples, so 400 columns per chunk for the final set and 100 columns per chunk for the reduced set. Given each chunk represents an epistemic sample I would probably first compute the mean of each chunk for each pixel, then you only need to compute the two bounding CDFs rather than all of them

## Powerscale Paths:
Pulse 253:
P:\TaskFiles\AMT-0038\KeyChallenge4-Processed\KC4-HIVE-0003\KC4-STC-002\Exp01-DIC-IR01\Test07-1550A

Pulse 254:
P:\TaskFiles\AMT-0038\KeyChallenge4-Processed\KC4-HIVE-0003\KC4-STC-002\Exp01-DIC-IR01\Test08-1550A

Pulse 255:
P:\TaskFiles\AMT-0038\KeyChallenge4-Processed\KC4-HIVE-0003\KC4-STC-002\Exp01-DIC-IR01\Test09-1550A

## TODO

We want to compute validation metrics for the following:
- All working thermocouples
- Coil voltage
- All viable fields (not sure if we have reliable temperature field data)

Ideally these would include experimental epistemic error, which is recorded along with the aleatoric statistics here: STC2_1550A_SteadySummary.csv

Worth noting that I have used all 1550A pulses for sample 2 (pulse 253, 254, 255) as an amalgamated dataset for the probabilistic simulation and the experiment summary above. The DIC images to use for steady state in the experiments are recorded here: SteadyStateDICReference.csv
