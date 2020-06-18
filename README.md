[How Google does Machine Learning](https://www.coursera.org/learn/google-machine-learning)
==========================================================================================
![image](https://github.com/Sayak007/How-Google-Does-MAchine-Learning/blob/master/How%20Google%20does%20Machine%20Learning/certificate/image.png)
=================================================================================================================================================

What is machine learning, and what kinds of problems can it solve? Google thinks about machine learning slightly differently -- of being about logic, rather than just data. We talk about why such a framing is useful for data scientists when thinking about building a pipeline of machine learning models. 

Then, we discuss the five phases of converting a candidate use case to be driven by machine learning, and consider why it is important the phases not be skipped. We end with a recognition of the biases that machine learning can amplify and how to recognize this.

### Intro to Specialization

* In 5 years -- 2012-17 Google has built and deployed over 4000 ML models
* Google Clould offers great tools and services for deploying ML models to prodiction
* Goals
  * ML with TensorFlow
  * Improving ML accuracy
  * ML at scale
  * Specialized ML models

Refs: [Graffiti Artist Classifier](https://cloud.google.com/blog/products/ai-machine-learning/who-street-artist-building-graffiti-artist-classifier-using-automl), [Pose-Estimator with Move Mirror](https://www.blog.google/technology/ai/move-mirror-you-move-and-80000-images-move-you/)

Labs and demos: [Training Data Analyst](https://github.com/GoogleCloudPlatform/training-data-analyst)

### What it means to be AI-first

* Artificial Intelligence is a discipline; machine learning is a specific way of solving AI problems
* In ML, machines learn. They don’t start out intelligent, become intelligent.
* Train an ML model with examples, then predict with a trained model
* Neural networks is one important technology we use
* ML replaces heuristics, it converts examples into knowledge
* Many ML projects fail because of training-serving skew

### How Google does ML

* Google infuses ML into almost all its product
* The ML surprise
  * Defining KPI’s
  * Collecting data
  * Building infrastructure
  * Optimizing ML algorithm
  * Integration
* Path to ML: The 5 phases
  * Individual contributor
  * Delegation
  * Digitization
  * Big Data and Analytics
  * Machine learning

### Inclusive ML

* Machine learning and human bias
* The confusion matrix leads to evaluation metric insights
  * True positives
  * False positives -- Type I error
  * False negatives -- Type II Error
  * True negatives
* Sometimes false negatives are better than false positives
* The _Equality of Opportunity_ approach strives to give individuals an equal chance of desired outcome
  * Simulating decisions with no constraints can lead to unequal distribution
  * Simulating decisions with group unaware holds everyone to the same standard, which can be unfair to some groups
* How to find errors in your dataset using Facets
  * Gives users a quick understanding of the distribution of values across features of their datasets

### Python notebooks in the cloud

* AI Platform Notebooks (formerly Cloud Datalab) are a fully hosted version of the popular JupyterLab notebook environment
* Compute Engine and Cloud Storage
  * Customizable machine types and flexible compute options
  * Control latency and availability with zones and regions

```bash
$ datalab create my-datalab-vm --machine-type n1-highmem-8 --zone us-central1-a
```

Lab: [Geographic data in Datalab](https://github.com/GoogleCloudPlatform/datalab-samples/blob/master/basemap/earthquakes.ipynb)

* Analyzing data using AI Platform Notebooks and BigQuery

Lab: [Data Analysis using Datalab and BigQuery](https://googlecoursera.qwiklabs.com/focuses/35205), [(notebook)](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/01_googleml/data_analysis.ipynb)

```python
query="""
SELECT
  departure_delay,
  COUNT(1) AS num_flights,
  APPROX_QUANTILES(arrival_delay, 10) AS arrival_delay_deciles
FROM
  `bigquery-samples.airline_ontime_data.flights`
GROUP BY
  departure_delay
HAVING
  num_flights > 100
ORDER BY
  departure_delay ASC
"""

from google.cloud import bigquery
df = bigquery.Client().query(query).to_dataframe()
df.head()

import pandas as pd
percentiles = df['arrival_delay_deciles'].apply(pd.Series)
percentiles = percentiles.rename(columns = lambda x : str(x*10) + "%")
df = pd.concat([df['departure_delay'], percentiles], axis=1)
df.head()

without_extremes = df.drop(['0%', '100%'], 1)
without_extremes.plot(x='departure_delay', xlim=(-30,50), ylim=(-50,50));
```

* Machine Learning APIs
  * [Cloud Vision](cloud.google.com/vision) - Complex image detection with a simple REST request
  * Cloud Video Intelligence - Understands your video entities by shot, frame or video level
  * Cloud Speech - Speech to text transcription in 100+ languages
  * Cloud Translation - Translate text into 100+ languages
  * [Cloud Natural Language](cloud.google.com/natural-language) - Understand text with a simple REST API request

Lab: [Invoking Machine Learning APIs](https://googlecoursera.qwiklabs.com/focuses/35206), [(notebook)](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/CPB100/lab4c/mlapis.ipynb)

### Summary
