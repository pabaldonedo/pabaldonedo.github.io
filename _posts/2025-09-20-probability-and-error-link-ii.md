---
title: The link between the probability realm and error minimisation (II)
date: 2025-09-20 14:08:00 +0200
categories: [Metrics, Insights]
tags: [regularization, probability, log-likelihood, error, metrics]     # TAG names should always be lowercase
author: pabaldonedo
description: Insights on the relation of L1 & L2 regularization and bayesian probability
math: true
---

In our previous post, we discussed why probability is central to machine learning and how log-likelihood connects to mean squared error. In this follow-up, we’ll dive deeper into the link between Bayesian probability and L2 regularization, also known as ridge regression.

This post contains quite a bit of mathematical derivation. If you prefer just the main takeaways, feel free to jump directly to the Conclusion section. But before that, if Bayesian statistics is new to you, you might want to skim through the Bayes Rule Brush-Up first.


## Bayes rule brush up
If you are familiar with the Bayes Rule you may want to skip this section altogether. But if this is new to you, let's do super high level run over it.


At its core, Bayesian statistics is about handling uncertainty. Instead of assigning a single “true” value to a variable or parameter, we describe it using a probability distribution.

For example, instead of saying:

> “The average male height in country X is 175 cm.”

A Bayesian approach would be:

> “Based on our data, the average height is most likely between 173 cm and 177 cm.”


Where do these ranges come from? 

Where do these ranges come from? From an iterative process involving two components:
 - **Prior:** What we already believe or expect.
 - **Likelihood:** What the data tells us.

Combining both we update our belief forming what is known as the _posterior_. So the prior acts like a mediation not to believe everything you see, if it is very rare, or to require less data points to draw conclusions when it is in line with expectation. 


Think about it in everyday life:

If something strongly contradicts your expectations, you won’t believe it immediately—you’ll want more evidence. But you do not disregard it enterily, you might now be more open to consider what you observed as a posibility in the future.

If something aligns with what you already expect, you’ll accept it more quickly.

This is exactly how priors (expectations) and likelihoods (data) interact.


Mathematically, this uncertainty is represented via probability distributions. And the update process via the following equation

$$
\begin{equation}
P(y|X) = \frac{P(y) P(X | y)}{P(X)} = \frac{prior \cdot likelihood}{evidence}
\label{eq:bayes_rule}
\end{equation}
$$

Here:

- **Posterior** = $$P(y$$\|$$X)$$ what we want to know (updated belief).
- **Prior** = $$P(y)$$ what we believed before seeing the data.
- **Likelihood** = $$P(X$$\|$$y)$$ how well the data fits our hypothesis.
- **Evidence** = $$P(X)$$a normalization term to ensure probabilities sum to 1.

Since the denominator is just a scaling factor, we often simplify to:

$$
\begin{equation}
P(y|X) \propto P(y) P(X | y)
\label{eq:bayes_rule_proportion}
\end{equation}
$$

Where $$\propto$$ means _proportional to_.

## Bayes and L2 regularization

Let’s recall the setup from previous post:
$$
\begin{equation}
y = f(X, \theta) + \epsilon \qquad \epsilon \sim \mathcal{N}(0, \sigma^2)
\label{eq:input_output_relationship}
\end{equation}
$$
Where:
 - **X**: features
 - $$\boldsymbol{\theta}$$: parameters (the ones we’re trying to learn)
 - **y**: target label

X and y are measured, but $$\theta$$ are the parameters. Now, since we are in the bayesian realm, parameters are not fixed numbers—they also have distributions!

Let's assume that our expectation (prior) for the parameter follows a normal distribution like this:


$$
\begin{equation}
p(\theta) \sim \mathcal{N}(\theta_0, \frac{1}{\sqrt{2\lambda}})
\label{eq:weights_prior}
\end{equation}
$$

Where:
 - $$\boldsymbol{\theta_0}$$ the average value we expect the weights to take
 - $$\boldsymbol{\lambda}$$ a hyperparameter controlling how strongly we believe this expectation, i.e. the variance


Now let's derive define the likelihood. From $$\ref{eq:input_output_relationship}$$, $$f(X, \theta)$$ is fixed for a given $$\theta$$ only having probabilities coming from $$\epsilon$$. So we end up with:

$$
\begin{equation}
p(y | \theta, X) \sim \mathcal{N}(f(X, \theta_0), \sigma^2)
\label{eq:weights_posterior}
\end{equation}
$$

Putting it all together, Bayes’ rule gives the posterior for θ:

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

So we end up with:

$$
\begin{equation}
\max_\theta \log p(\theta | y, X)  = \min_\theta ||y_i - f(x_i, \theta)||^2 + \lambda ||\theta - \theta_0||^2
\end{equation}
$$

The same mean square error as in the previous post, plus the additional term $$ \lambda \| \theta - \theta_0\|^2 $$. This is the L2 regularization term. You may usually see it with $$\theta_0 = 0$$ simplifying to $$ \lambda \| \theta\|^2 $$

## Bayes and L1 regularization

Just like we did with the Gaussian prior (which gave us L2 regularization), we can try a different assumption for our weights. This time, let’s use the [Laplace distribution]( https://en.wikipedia.org/wiki/Laplace_distribution):

$$
\begin{equation}
p(\theta) \sim \mathcal{Laplace}(\theta_0, \frac{1}{b}) =  \frac{b}{2} e^{-b |\theta - \theta_0|}
\label{eq:l1_weights_prior}
\end{equation}
$$

Here:
- $$\boldsymbol{\theta_0}$$ = the location (our prior guess for the weight value).
- $$\boldsymbol{\frac{1}{b}}$$ = the scale (controls how tightly concentrated the prior is around $$\theta_0$$).

If we take the log-posterior as before, but now with a Laplace prior, we get:

$$
\begin{equation}
\begin{aligned}
&\max_\theta \log p(\theta | y, X) 
\\& = \max_\theta \log p(\theta) p (y| \theta, X)
\\& = \max_\theta  \log p(\theta) + \log p (y| \theta, X)
\\& = \max_\theta  \log \mathcal{Laplace}(\theta_0, \frac{1}{b}) - ||y_i - \theta_0||^2
\\& = \max_\theta \log \frac{b}{2} - b |\theta - \theta_0| - ||y_i - f(x_i, \theta)||^2
\\& = \min_\theta ||y_i - f(x_i, \theta)||^2 +  b |\theta - \theta_0|
\end{aligned}
\end{equation}
$$

Which is the L1 or LASSO regularization!

## Conclusion

We’ve now seen how regularization is equivalent to the posterior, in bayes terms, of the weights when we assume different prior distributions:
 - L2 regularization (ridge regression): arises from assuming a Gaussian prior on the weights, centered at $$\theta_0$$.
 - L1 regularization (lasso): arises from assuming a Laplace prior, which has a sharp peak at $$\theta_0$$ and heavier tails.

In plain English:

- Regularization is just a way of saying, _We believe our parameters $$\theta$$ should be around $$\theta_0$$._
- The parameter $$\lambda$$ controls how stubborn we are about this belief.
  - A large $$\lambda$$ = we strongly believe $$\theta$$ must stay near $$\theta_0$$, so we need a lot of evidence (data) to move away.
  - A small $$\lambda$$ = we are more flexible with our beliefs and willing to let the data pull $$\theta$$ away from $$\theta_0$$.

And that’s the Bayesian intuition behind why ridge (L2) and Lasso (L1) regressions work.
