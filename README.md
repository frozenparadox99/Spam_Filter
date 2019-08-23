
# Notebook Imports


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
```

# Constants


```python

TOKEN_SPAM_PROB_FILE='SpamData/03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE='SpamData/03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE='SpamData/03_Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX='SpamData/03_Testing/test-features.txt'
TEST_TARGET_FILE='SpamData/03_Testing/test-target.txt'

VOCAB_SIZE=2500
```

# Load the Data


```python
X_test=np.loadtxt(TEST_FEATURE_MATRIX,delimiter=' ')
y_test=np.loadtxt(TEST_TARGET_FILE,delimiter=' ')
prob_token_spam=np.loadtxt(TOKEN_SPAM_PROB_FILE,delimiter=' ')
prob_token_ham=np.loadtxt(TOKEN_HAM_PROB_FILE,delimiter=' ')
prob_all_tokens=np.loadtxt(TOKEN_ALL_PROB_FILE,delimiter=' ')
```

# Calculate the joint probability


```python
X_test.dot(prob_token_spam).shape
```




    (1724,)



### Set the prior


```python
PROB_SPAM=0.31113
```


```python
np.log(prob_token_spam)
```




    array([ -4.42105189,  -5.26513947,  -5.00014881, ..., -10.1608448 ,
           -11.41360777,  -9.6218483 ])




```python
joint_log_spam=X_test.dot(np.log(prob_token_spam)-np.log(prob_all_tokens))+np.log(PROB_SPAM)
```


```python
joint_log_spam[:5]
```




    array([22.42364417,  1.97853165, 17.82891956, 16.80986334, 19.60317473])




```python
joint_log_ham=X_test.dot(np.log(prob_token_ham)-np.log(prob_all_tokens))+np.log(1-PROB_SPAM)
```


```python
joint_log_ham[:5]
```




    array([-58.9920736 , -10.86575417, -34.76777607, -58.5930807 ,
           -53.13258963])




```python
joint_log_spam.size
```




    1724



# Making Predictions


```python
prediciton=joint_log_spam>joint_log_ham
```


```python
prediciton[-5:]
```




    array([ True, False, False, False, False])




```python
prediciton[:5]
```




    array([ True,  True,  True,  True,  True])




```python
y_test[:5]
```




    array([1., 1., 1., 1., 1.])



# Metrics and Evaluation


```python
correct_doc=(y_test==prediciton).sum()
print(correct_doc)
```

    1685
    


```python
numdocs_wrong=X_test.shape[0]-correct_doc
print(numdocs_wrong)
```

    39
    


```python
accuracy=correct_doc/len(X_test)
print(accuracy)
```

    0.9773781902552204
    

# Simplification


```python
joint_log_spam2=X_test.dot(np.log(prob_token_spam))+np.log(PROB_SPAM)
joint_log_ham2=X_test.dot(np.log(prob_token_ham))+np.log(1-PROB_SPAM)
```


```python
prediciton2=joint_log_spam2>joint_log_ham2
```


```python
correct_doc=(y_test==prediciton2).sum()
print(correct_doc)
```

    1685
    


```python
accuracy=correct_doc/len(X_test)
print(accuracy)
```

    0.9773781902552204
    

# Visualising the Result


```python
yaxis_label='P(X | Spam)'
xaxis_label='P(X | NonSpam)'

linedata=np.linspace(start=-2000,stop=500,num=500)
```


```python
plt.figure(figsize=(11,7))
plt.xlabel(xaxis_label,fontsize=14)
plt.ylabel(yaxis_label,fontsize=14)



plt.scatter(joint_log_ham,joint_log_spam)
plt.show()
```


![png](output_32_0.png)


## The Descision Boundary


```python
plt.figure(figsize=(11,7))
plt.xlabel(xaxis_label,fontsize=14)
plt.ylabel(yaxis_label,fontsize=14)

plt.xlim([-2000,500])
plt.ylim([-2000,500])

plt.scatter(joint_log_ham,joint_log_spam,alpha=0.5,s=25)
plt.plot(linedata,linedata,color='orange')
plt.show()
```


![png](output_34_0.png)



```python
plt.figure(figsize=(16,7))

plt.subplot(1,2,1)

plt.xlabel(xaxis_label,fontsize=14)
plt.ylabel(yaxis_label,fontsize=14)

plt.xlim([-2000,500])
plt.ylim([-2000,500])

plt.scatter(joint_log_ham,joint_log_spam,alpha=0.5,s=25)
plt.plot(linedata,linedata,color='orange')

plt.subplot(1,2,2)

plt.xlabel(xaxis_label,fontsize=14)
plt.ylabel(yaxis_label,fontsize=14)

plt.xlim([-200,60])
plt.ylim([-200,60])

plt.scatter(joint_log_ham,joint_log_spam,alpha=0.5,s=3)
plt.plot(linedata,linedata,color='orange')


plt.show()
```


![png](output_35_0.png)



```python
sns.set_style('whitegrid')
labels='Actual Category'

summary_df=pd.DataFrame({yaxis_label:joint_log_spam,xaxis_label:joint_log_ham,labels:y_test})
```


```python
sns.lmplot(x=xaxis_label,y=yaxis_label,data=summary_df,size=6.5)
plt.show()
```

    C:\Users\Sushant Lenka\.conda\envs\machineLearning\lib\site-packages\seaborn\regression.py:546: UserWarning: The `size` paramter has been renamed to `height`; please update your code.
      warnings.warn(msg, UserWarning)
    


![png](output_37_1.png)



```python
plt.figure(figsize=(11,7))
plt.xlabel(xaxis_label,fontsize=14)
plt.ylabel(yaxis_label,fontsize=14)



plt.scatter(joint_log_ham2,joint_log_spam2)
plt.show()
```


![png](output_38_0.png)



```python
sns.set_style('whitegrid')
labels='Actual Category'

summary_df=pd.DataFrame({yaxis_label:joint_log_spam2,xaxis_label:joint_log_ham2,labels:y_test})
linedata2=np.linspace(start=-14000,stop=1,num=1000)
```


```python
sns.lmplot(x=xaxis_label,y=yaxis_label,data=summary_df,height=6.5,fit_reg=False,scatter_kws={'alpha':0.5,'s':25})
plt.xlim([-2000,1])
plt.ylim([-2000,1])

plt.plot(linedata,linedata,color='black')
plt.show()
```


![png](output_40_0.png)



```python
sns.lmplot(x=xaxis_label,y=yaxis_label,data=summary_df,height=6.5,fit_reg=False,scatter_kws={'alpha':0.5,'s':25},hue=labels,
          markers=['o','x'],palette='hls',legend=False)
plt.xlim([-500,1])
plt.ylim([-500,1])

plt.plot(linedata,linedata,color='black')

plt.legend(('Decision Boundary','Nonspam','Spam'),loc='lower right',fontsize=14)
plt.show()
```


![png](output_41_0.png)


### False Positives and False Negatives


```python
np.unique(prediciton,return_counts=True)
```




    (array([False,  True]), array([1138,  586], dtype=int64))




```python
true_pos=(y_test==1) & (prediciton==1)
```


```python
true_pos.sum()
```




    568




```python
false_pos=(y_test==0) & (prediciton==1)
```


```python
false_pos.sum()
```




    18




```python
false_neg=(y_test==1)&(prediciton==0)
```


```python
false_neg.sum()
```




    21




```python
true_neg=(y_test==0)&(prediciton==0)
```


```python
true_neg.sum()
```




    1117



## The Recall Metric


```python
recall_score=(true_pos.sum())/(true_pos.sum()+false_neg.sum())
print(recall_score)
```

    0.9643463497453311
    

## The Precision Metric


```python
precision_score=(true_pos.sum())/(true_pos.sum()+false_pos.sum())
print(precision_score)
```

    0.9692832764505119
    

## The F1 Metric 


```python
f1_score=(2*precision_score*recall_score)/(precision_score+recall_score)
print(f1_score)
```

    0.9668085106382979
    


```python

```
