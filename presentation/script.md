# Robust Station Selection for Microtransit Under Demand Uncertainty

## What is Microtransit?

So firstly, what is microtransit?

Microtransit seeks to fill the gap between existing services such as Private Hire Vehicles and Public Transit.

It aims to be a flexible on-demand service that can be hailed by any member of the public.

## Problem Setting

Thus, for the microtransit service, it is important to select these stations, termed "Virtual Bus Stops".

So that the routing of vehicles is efficient and service coverage is sufficient.

A concern we would then have is that a station selection solution that is based on nominal demand might not provide appropriate coverage if demand shifts.

## Goal

Build a optimization model for station selection

and we want to protect against demand deviations

## Two-Stage Model (could skip)

So to establish some components of the model, we can focus on the variables z_js where we choose specific stations to select in each "scenario", think times like morning, afternoon, evening, night.

We then want to minimize for each origin-destination, the walking distance to the station and the routing cost between the assigned stations.

Each customer also has a limit to how far they are willing to walk, which results in this A\_{od} set of stations

## Nominal Objective and Constraints

We arrive at our model which minimizes the weighted sum of the routing and walking cost of the assignments, where the weights are the amount of demand.

Each assignment can only be performed once, and the assignments are subject to the stations being activated and available.

We also restrict the amount of stations that can be built.

## Demand Uncertainty Set

So to protect against the uncertainty demand, we create this kind of budget uncertainty set.

We assume every origin-destination has a lower and upper bound in demand and that the total amount of demand experienced is capped at this upper amount Q.

The idea here is that from day to day, the demand patterns can shift, but since it is unusual for the population of the system to increase, we can cap the total amount to upper case Q.

Another thing to note here is that while demand is technically integer-valued, we might have to use some other more complex methods to solve the robust problem but since the bounds are integer, as well as the terms of the assignment being linear, we can simply relax the integer constraints.

## Robust Counterpart

So to construct the robust counterpart, we first recognise that the objective has a simple recourse function where $x$ is simply choosing the lowest $a$ cost.

This means we can bring the $q_{ods}$ coefficient out of the sum and take the dual of the max problem to attain the following robust objective and the additional constraints.

## Calibration of q and Q

For the calibration of the uncertainty set, the data was aggregated based on each scenario and for the upper bound values we will simply take a certain quantile value of the demand of the OD pair and also the upper bound of the overall demand, for upper case Q

## Experiment Instance

For the experiment, we then test the optimization over this zone in zhu zhou china. The data is from its microtransit service and we opt to take the first 3 days of april as our "train" dataset and test it on the assignment in the whole month of may.

## Data and Experimental Design

These are some further calibration values:

The number of station activations in each scenario were varied and we also tested different quantile values to calibrate the uncertainty set

## Sampled-Network Results

Note that in these experiments, the weightage of the routing cost is relatively high at $10$.

We see that the nominal model almost always has better cost but the spread of the cost is lower with the robust model.

Interestingly enough, the coverage of the nominal model is perfect and all the orders in the out of sample set are able to find appropriate assignments.

## Lack of Data Coverage

Thus, I decided to try making the demand data even more sparse to give the robust model an edge and see if it performs as expected when demand data coverage is low.

## Main Sparse-Network Results

So as expected, the nominal model did not have full covergae of the out of sample data but the robust models did.

The trend still holds that robust models did not find the lowest cost solution in the out of sample data but rather it still had the lowest spread of the cost values.

## Cost CDFs

Turning to the CDFs of the cost in specific scenarios, we would hope to find a CDF more like the right side where it might be tough to see but at the top, the robust values are just abit better at higher percentiles than the nominal model.

But mostly we find the result on the left that the decrease in spread usually just resulted in a higher cost for people at the lower quantiles.

## Activations

This can also be seen from the specific activated stations, we see that on the right the robust solution spreads out the activations a lot more which affects the cost negatively as compared to the nominal model.

## Conclusion

Overall, the budget uncertainty set helps model the uncertain demand.

And we find out that robustness guarantees coverage but we could also do so through having explicit constraints

In the future, we could explore more complex recourse functions and the current cost function might be overly simplistic

and clearly in this case where demand is quite sufficient, the robust model does not shine as much here, and there will be situations where it will be more effective.
