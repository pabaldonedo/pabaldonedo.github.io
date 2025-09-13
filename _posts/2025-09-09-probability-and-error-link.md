---
title: The link between the probability realm and error minimisation (I)
date: 2025-09-09 14:08:00 +0200
categories: [Metrics, Insights]
tags: [regularization, probability, log-likelihood, error, metrics]     # TAG names should always be lowercase
author: pabaldonedo
description: Insights on the relation of mean square error and log likelihood
math: true
---

Many students, including myself, start their journey in Data Science with courses like Andrew Ng's famous [Machine Learning Course](https://www.deeplearning.ai/courses/machine-learning-specialization/) on Coursera. Very early on, they encounter an equation like this:


$$
\begin{equation}
  J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2} (h_{\theta}(x^{(i)}) - y^{(i)})^2
  \label{eq:mse_andrew}
\end{equation}
$$

This is the **Mean Squared Error (MSE)** loss. We are told that training a model means finding parameters $$\theta$$ that minimise this loss.

So far, so good.

But then, when reading research papers or more advanced texts, a different expression appeared:


$$
\begin{equation}
  \text{Max Log Likelihood} = \max_{\theta} p(y | x, \theta)
  \label{eq:max_likelihood}
\end{equation}
$$


Where does that probability come from?
I remember feeling very confused: _I was minimising squared errors, not dealing with probabilities. How are these connected?_


In this series of posts, we seek to bridge that gap.

- In this post, we’ll see why probability naturally arises in machine learning and how MSE and maximum log-likelihood are actually two sides of the same coin.

- In the next post, we’ll explore how regularisation connects with Bayesian probability.

-----

## Why probabilities belong in machine learning

### Machine learning project steps

Every machine learning project starts with collecting some data. Data may come from:
 - Nature (e.g. temperatures, rainfall, gene expression etc.)
 - Business problem (e.g. user churn, product quality, pricing, etc.).

The end goal is always the same: **build a model, i.e. a mathematical function, that captures the process generating the data**. This is known as solving the _inverse problem_.

As an example, imagine an online marketplace wants to predict whether a user is interested in a new perfume:
 - Users who are interested will likely spend more time browsing perfumes and related products.
 - Users who are not interested will behave differently.

The task of machine learning here is to reverse the data generation process: from a user’s data (clicks, purchases, time spent) infer whether they are interested or not.

<img src="/assets/img/ml_project/data_generation_process_with_text.png" width="100%" alt="machine learning project steps" />


### A probabilistic lens on prediction

Let's deep dive with another example: house pricing. Suppose we want to predict house prices from house features (size, floor, location…). Let’s start simple:

$$
\begin{equation}
  price = \log(size) \\
  \label{eq:simplest_pricing}
\end{equation}
$$

This would mean that for a given size, the price is always fixed. But reality is messier: two houses of the same size will not always cost the same.

So a more realistic model is:

$$
\begin{equation}
  price = \log(size) + \delta + \epsilon
  \label{eq:simplest_pricing_with_errors}
\end{equation}
$$

- $$\delta$$ = the effect of other factors we ignored (location, floor, etc.).
- $$\epsilon$$ = noise (e.g. data collection errors, seller’s mood, unmeasurable quirks).


Note that even if we considered all factors that determine the price (i.e. $$\delta = 0$$), randomness $$\epsilon$$ would still remain. We will assume $$\delta =0$$ for clarity in the rest of the post.

Let's assume noise $$\epsilon$$ follows a Normal distribution of zero mean and $$\sigma$$ variance, which is a common scenario:

$$
\begin{equation}
\epsilon \sim \mathcal{N}(0, \sigma^2)
\label{eq:noise_normal_distribution}
\end{equation}
$$

We reach an interesting observation: **the true price is not a single value but follows a probability distribution around our price prediction curve.**  The figure below illustrates this idea:

<img src="/assets/img/probability/log_surface_price_rotated.svg" width="100%" alt="probability surface over log curve"/>

Mathematically:

$$
\begin{equation}
  p(y) = \mathcal{N}(f(x), \sigma^2)
  \label{eq:ml_prob_prediction}
\end{equation}
$$

- $$f(x)$$ is the model output
- $$\sigma^2$$ is the variance of the noise



Let's pause and ponder for a moment: **our model is really about predicting the center of a probability distribution, not just a single number.**

Note: the distribution could be of any family but in our example, and the common case, is a normal distribution.

The training goal becomes: **find the parameters that make the observed data most probable**. Therefore, we adjust the parameters with our dataset:

$$
\begin{equation}
  \max p(Y) \qquad \qquad \text{Where } p \sim \mathcal{N}(f(x), \sigma^2)
  \label{eq:ml_max_likelihood}
\end{equation}
$$


## Deriving the optimisation problem over datasets

In the previous section we conclude that we want to maximise the probability over our dataset. Usually, we assume that all the samples are Independent and Identically Distributed (I.I.D.) so the probability of the entire dataset is equal to multiply each individual probabilities:

$$
\begin{equation}
  \max p(Y) = \max \prod_i p(y_i) \qquad \qquad \text{Where } p \sim \mathcal{N}(f(x_i), \sigma^2)
  \label{eq:ml_max_likelihood_dataset}
\end{equation}
$$

Taking the log we can simplify products into sums:

$$
\begin{equation}
  \max \log p(Y) = \max \sum_i \log p(y_i) \qquad \text{Where } p \sim \mathcal{N}(f(x_i), \sigma^2)
  \label{eq:ml_max_log_likelihood_dataset}
\end{equation}
$$

This is called the **log-likelihood**.

Let's now expand the terms under our normal assumption:

$$
\begin{equation}
  \begin{aligned}
  & \max \log p(Y) \\
  & = \max \sum_i \log p(y_i) \\
  & = \max_f \sum_i \log \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{||y_i - f(x_i)||^2}{2 \sigma^2}} \\
  & = \max_f \sum_i \log \frac{1}{\sqrt{2\pi \sigma^2}} - \frac{1}{2 \sigma^2} ||y_i - f(x_i)||^2 \\
  & = \max_f \sum_i - \frac{1}{2 \sigma^2} ||y_i - f(x_i)||^2
 \end{aligned}
  \label{eq:ml_max_log_likelihood_dataset_to_ml}
\end{equation}
$$

The last step we got rid of $$\log \frac{1}{\sqrt{2\pi \sigma^2}}$$ since it does not depend on $$f(x)$$ so it is a constant that does not affect on the maximisation problem.

Note the final result, does it sound familiar? Let's rewrite it a little bit further:

$$
\begin{equation}

\max_f \sum_i - \frac{1}{2 \sigma^2} ||y_i - f(x_i)||^2 \leftrightarrow \min_f \sum_i \frac{1}{2 \sigma^2} ||y_i - f(x_i)||^2
\end{equation}
$$

And there it is! We just derived derived Mean Squared Error** from maximum likelihood with a Gaussian noise assumption with $$\sigma = 1$$.


## The link

So what have we seen?

- **Minimising MSE is equivalent to maximising the log-likelihood under the assumption of Gaussian noise with unit variance.**

- Error minimisation and probability maximisation are not separate approaches: they are two perspectives on the same underlying principle.

This link often goes unmentioned in beginner resources, leaving students puzzled. But once you see it, you realise why probability naturally sits at the heart of machine learning. The goal of this post is to give this insight, and derivation, if you have not come accross it yet.

👉 In the next post, we’ll push this probabilistic view further, showing how L2 regularisation emerges naturally from Bayesian probability.
