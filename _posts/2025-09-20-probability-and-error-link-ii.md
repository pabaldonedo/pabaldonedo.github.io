---
title: The link between the probability realm and error minimisation (II)
date: 2025-09-20 14:08:00 +0200
categories: [Metrics, Insights]
tags: [regularization, probability, log-likelihood, error, metrics]     # TAG names should always be lowercase
author: pabaldonedo
description: Insights on the relation of L2 regularization and bayesian probability
math: true
---

In our last post we discuss why taking about probability in a Machine Learning Project make senses and the link between log likelihood and mean square error. This post elaborates further and goes through the link between bayesian probability and L2 regualization

This post contains mostly mathematical derivation. If you are just intersted in a one-liner jump to the conclusion section stopping by the bayes rule brush up if you are unfamiliar with bayesian statistics.

## Bayes rule brush up
If you are familiar with the Bayes Rule you may want to skip this section altogether. If this is new to you let's do super high level run over it.

The very foundation of bayesian statistics is about dealing with uncertainty. This translates to every variable or parameter has a distribution of likely values it can take, instead of a single value. For example, the variable _average male height in country X_ would not be `175 cm` but `from our observations the value ranges between 173 and 177 cm`.

But, where does those ranges come from? The way bayesian comes to that conclusion is in an iterative process where each step make use of 2 components:
 - *Prior*
 - *Likelihood*

The prior can either be any previous knowledge we have or any expectation we may have. On the other hand, the likelihood is the knowledge we get from actual data we observe. Combining both we update our knowledge forming what is known as the _posterior_. So the prior acts like a mediation not to believe everything you see, if it is very rare, or to require less data points to draw conclusions when it is in line with expectation. 

Think it in your day to day. If you observe something that goes heavily against your expectations you are less willing to believe it 100% and you may want to have further data to validate the observations. But you do not disregard it enterily, you might now be more open to consider what you observed as a posibility in the future.


Mathematically, this uncertainty is represented via probability distributions. And the update process via the following equation

$$
\begin{equation}
posterior = P(y|X) = \frac{P(y) P(X | y)}{p(X)} = \frac{prior \cdot likelihood}{evidence}
\label{eq:bayes_rule}
\end{equation}
$$

The denominator acts as a normalization factor so that everything sum up to 1 as a probability must. That is why in many cases it is obviated in derivations and the final result is just then normalised to sum up to 1. So we can just focus on:


$$
\begin{equation}
P(y|X) \propto P(y) P(X | y)
\label{eq:bayes_rule_proportion}
\end{equation}
$$

Where $$\propto$$ means _proportional to_.

## Bayes and regularization

Let's recover the equations from the previous post:

$$
\begin{equation}
y = f(X, \theta) + \epsilon \qquad \epsilon \sim \mathcal{N}(0, \sigma^2)
\label{eq:input_output_relationship}
\end{equation}
$$
Where:
 - X: features
 - $$\theta$$: parameters
 - y: label


X and y are measured, but $$\theta$$ are the parameters. Since we are in the bayesian realm, every parameter has a distribution! Let's assume that our expectation (prior) for the parameter follows a normal distribution like this:


$$
\begin{equation}
p(\theta) \sim \mathcal{N}(\theta_0, \frac{1}{\sqrt{2\lambda}})
\label{eq:weights_prior}
\end{equation}
$$

Where:
 - $$\theta_0$$ the average value we expect for the weight
 - $$\lambda$$ a hyperparameter controlling the variance


We also need to define the likelihood. From $$\ref{eq:input_output_relationship}$$, if we fix the parameters $$\theta$$, we have $$f(X, \theta)$$ fixed for a given $$X$$ and a probability distribution coming from $$\epsilon$$. So we end up with:


$$
\begin{equation}
p(y | \theta, X) \sim \mathcal{N}(f(X, \theta_0), \sigma^2)
\label{eq:weights_posterior}
\end{equation}
$$


Let's put altogether and we get the posterior for our weights:

$$
\begin{equation}
p(\theta | y, X) \propto p(\theta) p (y| \theta, X) = \mathcal{N}(\theta_0, \frac{1}{\sqrt{2\lambda}})  \mathcal{N}(f(X, \theta_0), \sigma^2)
\end{equation}
$$

Let's maximisize the log of this:


$$
\begin{equation}
\begin{aligned}
&\max_\theta \log p(\theta | y, X) 
\\& = \max_\theta \log p(\theta) p (y| \theta, X)
\\& = \max_\theta  \log p(\theta) + \log p (y| \theta, X)
\\& \stackrel{*}{=} \max_\theta  \log\mathcal{N}(f(X, \theta_0), \sigma^2) - ||y_i - \theta_0||^2
\\& = \max_\theta \log  \frac{1}{\sqrt{2\pi 2 \lambda}} e^{-\frac{||\theta - \theta_0||^2}{2 \frac {1}{2\lambda}}}  - ||y_i - f(x_i, \theta)||^2
\\& = \max_\theta \log \frac{1}{\sqrt{4 \pi \lambda}} - \lambda ||\theta- \theta_0||^2 - ||y_i - f(x_i, \theta)||^2
\\& = \min_\theta ||y_i - f(x_i, \theta)||^2 + \lambda ||\theta - \theta_0||^2
\end{aligned}
\end{equation}
$$


\* We just use the results of the previous post for $$\log p (y$$ \| $$\theta, X)$$  for the second term in the sum

So the end result is:


$$
\begin{equation}
\max_\theta \log p(\theta | y, X)  = \min_\theta ||y_i - f(x_i, \theta)||^2 + \lambda ||\theta - \theta_0||^2
\end{equation}
$$

The same mean square error as in the previous post, plus the additional term $$ \lambda \| \theta - \theta_0\|^2 $$. This is the L2 regularization term. You may usually see it with $$\theta_0 = 0$$ so simply as $$ \lambda \| \theta\|^2 $$

## Conclusion

We have seen that L2 regularization is equivalent to the posterior, in bayes terms, of the weights with a normal prior with mean $$\theta_0$$ and variance $$\frac{1}{\sqrt{2\lambda}}$$.

In more intuitive terms, this means that regularization is equivalent to set our expectations on the weights to have a value around $\theta_0$ controlling how much it can deviates from it with the $\lambda$ parameter. High values of $$\lambda$$ translate in small variance, i.e. we have a very strong opinion that the weight $$\theta$$ is very close to $$\theta_0$$ and we would need a ton of data evidence to consider a different value. Contrary, a small $$\lambda$$ means that we believe $$\theta$$ is around $$\theta_0$$ but we are willing to change our minds easily if the data suggest otherwise.

And there we have it: the link between ridge regression and bayes probability!
