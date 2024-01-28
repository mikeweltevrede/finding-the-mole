# Introduction
I am using the book "Bayesian Modeling and Computation in Python"<sup>1</sup>
([link to online book](https://bayesiancomputationbook.com/welcome.html#)). This book also gives a good introduction to
certain concepts in Bayesian modelling, so I recommend checking it out

<sup>1</sup> Martin Osvaldo A, Kumar Ravin; Lao Junpeng Bayesian Modeling and Computation in Python Boca Ratón, 2021. ISBN 978-0-367-89436-8

# Modelling approach
In short, I want to apply a Bayesian approach for finding the Mole because it fits our use case well: after each episode
we gain more information, and we update our beliefs based on what we saw. If a person was in a group during a task where
we suspect the Mole would want to be, then we would tend to believe that they are more likely to be the Mole. As such,
the inference results after each episode serve as input for the inference for the next episode.

At the start, of course, we have no information and the prior distribution has to be uniform. One can make assumptions,
of course, such as whether they think the Mole is more likely to be male or female; based on previous years, one might
think that they probability that they'd choose 3 male Moles in a row is low. However, I want to base this model purely
on what groups I see people joining during the task and make no prior assumptions.
