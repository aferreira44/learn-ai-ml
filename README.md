# **learn**-ai-ml
Study material about AI and ML

## Artificial Intelligence

Design an intelligent agent that perceives its environmet and makes decisions to maximize chances of achieving its goal.

AI is the study of agents that perceive the world around them, form plans, and make decisions to achieve their goals. Its foundations include mathematics, logic, philoshopy, probability, linguistics, neuroscience and decision theory.

AI as the study of [intelligent agents](https://en.wikipedia.org/wiki/Intelligent_agent), human or not, that can perceive their environments and take actions to maximize their chances of achieving some goal.

Artificial Intelligence comprises all ML techniques, but it also includes other techniques such as search, symbolic reasoning, logical reasoning, statistical techniques that aren't deep learning based, and behavior-based approaches.

Subfields: vision, robotics, machine learning, natural language processing, planning, ...

**Goals**: 

- such as recognizing what's in a picture (computer vision, or more specifically, object recognition)

- converting recording of your voice into the words you meant (automated speech recognition, a subfield of natural language processing)

- finding the best way to get to grandmother's house (route planning).

- autonomy

- knowledge representation

- logical reasoning

- planning

- learning

- manipulating objects

- classifying documents

- and so on

**Techniques **(algorithms and data structures) we use in trying to achieve those goals) :

- supervised learning
- deep learning
- search
- planning
- reinforcement learning
- symbolic computation
- mathematical optimization
- and many others


Artificial intelligence is a broad set of software capabilities that make your software smarter. We think it's going to have as broad (and maybe broader) an impact on software as relational database technologies: It's hard to think of a company whose products or services you use today that aren't enabled by databases.

We're in the very early years of putting AI in all our software in the same way we put databases in all our software, and this trend will unfold over decades, not months or even years. AI is the new relational database, about to get into every important piece of software we write.

One way to think of what AI enables to is consider what it will make cheap and thereby ubiquitous. AI will make predictions cheap, true, but it will make other things cheap as well:

- Enabling things that move to drive or fly or sail themselves
- Understand people and objects and their relationships in the real world
- Optimize complex systems, such as driving patterns or electricity consumption in data centers
- Create content, such as newspaper articles, tweets, music, Websites, movie trailers, and eventually entire movies
- Understand people, help people understand software, and help people understand each other

Some people use Artificial Intelligence and Machine Learning interchangeably. In this guide, we'll treat Artificial Intelligence as the broadest term which includes all Machine Learning techniques, and we'll treat Deep Learning as subset of Machine Learning techniques.

---

- AI Effect
- Artificial Narrow Intelligence (ANI): can effectively perform a narrowly
  defined task.
- Artificial General Intelligence (AGI) - Strong AI: can successfully perform any  intellectual task that a human being can. E.g., learning, planning and decision-making under uncertainty, communicating in natural language, making jokes, manipulating people, trading stocks or reprogramming itself.
- Artificial Superintelligence (ASI)
- Strong / Weak AI ; Hard / Soft AI; Deep / Narrow AI
- Intelligence explosion
- Singularity

## Machine Learning

https://en.wikipedia.org/wiki/Machine_learning

ML is one of many subfields of AI, concerning the ways that computers learn from experience to improve their ability to think, plan, decide and act.

A ML algorithm enables it to identify patterns in observed data, build models that explain the world, and predict things without having explicit pre-programmed rules and models.

There are many different types of machine learning algorithms, including reinforcement learning, genetic algorithms, rule-based machine learning, learning classifier systems, and decision trees.

Gives "computers the ability to learn without being explicity programmed" (Arthur Samuel, 1959)

### Supervised Learning

The goal of supervised learning is to predict Y as accurately as possible when given new examples where X is known and Y is unknown.

  - Regression: predict a continuous numerical value. How much will that house sell for?

Regression predicts and estimate a value, a continuous target variable Y, based on input data X. Input data X includes all attributes that are relevant information in the data set. These attributes are called features, which can be numerical and categorical.

Target variable: the unknown variable we care about predictiong, and continuous means there aren't gaps (discontinuities) in the value that Y can take on. A person's weight and height are continuous values.

Discrete variables: Can only take on a finite number of values - for example, the number of kids somebody has.

Training data set: has labels, so your model can learn from these labeled examples.

Test data set: does not have labels. You don't know the value you're trying to predict. It's important that your model can generalize to situations it hasn't encountered before so that it can perform well on the test data.

  - Supervised Learning Algorithms
    - Linear Regression (Ordinary Least Squares OLS)
    	- Linear regression is a parametric method, which means it makes an assumption about the form of the function relating X and Y. Our goal is to learn the model parameters that minimize error in the model's predictions.

We have our data X, and corresponding target values Y. The goal of ordinary least squares (OLS) regression is to learn a linear model that we can use to predict a new y given a previously unseen x with as little error as possible.

To find the best parameters:
​	- Define a cost function, os loss function, that measures hpw inaccurate our model's predictions are.
​	- Find the parameters that minimize loss, i.e make our model as accurate as possible.
​	- Gradient descent: learn the parameters
​      - [scikit-learn](http://scikit-learn.org/stable/) and [TensorFlow](https://www.tensorflow.org/)

The goal of gradient descent is to find the minimum of our model's loss function by iteratively getting a better and better approximation of it.

![img](https://cdn-images-1.medium.com/max/873/0*XNjJ4YFPEF08f-1L.)

The function is f(β0,β1)=z. To begin gradient descent, you make some 
guess of the parameters β0 and β1 that minimize the function.

Next, you find the [partial derivatives](https://en.wikipedia.org/wiki/Partial_derivative) of the loss function with respect to each beta parameter: [*dz/dβ0, dz/dβ1*]. A **partial derivative** indicates how much total loss is increased or decreased if you increase *β0* or *β1* by a very small amount.

Put another way, how much would increasing your estimate of annual income assuming zero higher education (*β0*) increase the loss (i.e. inaccuracy) of your model? You want to go in the *opposite* direction so that you end up walking *downhill* and minimizing loss.

Similarly, if you increase your estimate of how much each incremental year of education affects income (*β*1), how much does this increase loss (*z*)? If the partial derivative *dz/β1* is a *negative* number, then *increasing* *β1* is good because it will reduce total loss. If it’s a *positive* number, you want to *decrease* *β1*. If it’s zero, don’t change *β1 because it means you’ve reached an optimum*.

Keep doing that until you reach the bottom, i.e. the algorithm **converged**  and loss has been minimized. There are lots of tricks and exceptional  cases beyond the scope of this series, but generally, this is how you  find the optimal **parameters** for your **parametric** model.

- Overfitting and underfitting: learning a function that perfectly explains the training data that the 
  model learned from, but doesn’t generalize well to unseen test data.

  - **Overfitting** happens when a model *overlearns* from the training data to the point that it starts picking up idiosyncrasies that aren’t representative of patterns in the real world. This becomes especially problematic as you make your model increasingly complex.

  - **Underfitting ** is a related issue where your model is not complex enough to capture the underlying trend in the data.

  - How to avoid them:

    - **Use more training data**. The more you have, the harder it is to overfit the data by learning too much from any single training example.

    - **Use regularization**.  Add in a penalty in the loss function for building a model that assigns  too much explanatory power to any one feature or allows too many  features to be taken into account.

      ![img](https://cdn-images-1.medium.com/max/873/1*rFT6mtU45diT0OJhlgDcBg.png)

      The first piece of the sum above is our normal cost function. The second piece is a **regularization term** that  adds a penalty for large beta coefficients that give too much  explanatory power to any specific feature. With these two elements in  place, the cost function now balances between two priorities: explaining  the training data and preventing that explanation from becoming overly  specific.

      The **lambda** coefficient of the regularization term in the cost function is a **hyperparameter:** a general setting of your model that can be increased or decreased (i.e. **tuned**) in  order to improve performance. A higher lambda value will more harshly  penalize large beta coefficients that could lead to potential overfitting. To decide the best value of lambda, you’d use a method  called **cross-validation**  which involves holding out a portion of the training data during  training, and then seeing how well your model explains the held-out  portion. We’ll go over this in more depth

- Bias and variance: Ultimately, in order to have a good model, you need one with low bias and low variance.

  - **Bias** is the amount of error introduced by approximating real-world phenomena with a simplified model.
  - **Variance** is how much your model's test error changes based on variation in the training data. It reflects the model's sensitivity to the idiosyncrasies of the data set it was trained on.

## Deep Learning

While deep learning is deservedly enjoying its moment in the sun, we're particularly excited about ensemble techniques that use a wide variety of machine learning and non-machine learning based approaches to solve problems. Google's AlphaGo program, for instance, uses Monte Carlo tree search in addition to convolutional neural networks (a special type of neural network) to guide the search process. We expect most autonomous driving systems to use traditional search techniques for route planning and deep learning for "safe path detection".

---

- Classification: assign a label. Is this a picture of a cat or a dog?
- Unsupervised learning: Clustering, dimensionality reduction, recommendation
- Reinforcement Learning: Reward maximization

----

## History

https://en.wikipedia.org/wiki/AI_winter

https://en.wikipedia.org/wiki/SHRDLU

https://en.wikipedia.org/wiki/ELIZA

https://en.wikipedia.org/wiki/History_of_artificial_intelligence

https://en.wikipedia.org/wiki/Dartmouth_workshop
