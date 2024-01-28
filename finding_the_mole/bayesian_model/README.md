# Introduction
I am using the book "Bayesian Modeling and Computation in Python"<sup>1</sup>
([link to online book](https://bayesiancomputationbook.com/welcome.html#)). This book also gives a good introduction to
certain concepts in Bayesian modelling, so I recommend checking it out.

# Modelling approach
In short, I want to apply a Bayesian approach for finding the Mole because it fits our use case well: after each episode
we gain more information, and we update our beliefs based on what we saw. If a person was in a group during a task where
we suspect the Mole would want to be, then we would tend to believe that they are more likely to be the Mole. As such,
the inference results after each episode serve as input for the inference for the next episode.

At the start, of course, we have no information and the prior distribution has to be non-informative<sup>2</sup>. One
can make assumptions, of course, such as whether they think the Mole is more likely to be male or female; based on
previous years, one might think that they probability that they'd choose 3 male Moles in a row is low. However, I want
to base this model purely on what groups I see people joining during the task and make no prior assumptions.

We want to model the "Mole probability" $Y$ for each participant given the groups they joined in the tasks. As such, we
define the following available parameters:
- $m_{i,t}$: 1 if person $i$ joined the "Mole group"<sup>3</sup> in task $t$, 0 otherwise. The vector over all
    participants is referred to as $m_t$.

This model as well as the parameter's prior distributions are taken from the [PyMC3 documentation](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html#a-motivating-example-linear-regression).
This is done due to still learning about how to code Bayesian models. In the future, these choices may differ based on
gained knowledge about the topic.

The model would look like:
```math
Y \sim N(\mu, \sigma^2)\
\mu = \alpha + \sum_{t} \beta_t m_t
```

And for the priors, we choose:
```math
\alpha \sim N(0, 100)\
\beta_t \sim N(0, 100)\
\sigma \sim |N(0,100)|
```

# Footnotes
<sup>1</sup> Martin Osvaldo A, Kumar Ravin, Lao Junpeng; Bayesian Modeling and Computation in Python; Boca Ratón, 2021.
ISBN 978-0-367-89436-8.

<sup>2</sup> Note that non-informative is not the same as using uniform priors. Using uniform priors is discouraged
[(Gelman \& Yao, 2020)](http://www.stat.columbia.edu/~gelman/research/unpublished/bayes_holes_2.pdf).

<sup>3</sup> See [data/README.md](../../data/README.md) for a definition of the "Mole group".
