## Multi-instance Learning with Transformer

For predicitve modeling, many real world use cases fall into the category of multi-instance learnings, such as medical image analysis, insurance claim photo analysis. In those use cases, the user may have multiple images taken/uploaded as input, and the model should make one final single decision as output. For examples, a customer who filed a accident claims usually upload many images, including the shot of the car, insurance policy, ids and other documents, to the insurance server as evidence, and the insurance decision are made based on one or more of those images. The instances can be in different modalities, such as videos, audios, images, and even strctured data.

In this multi-instance learning scenario, the model should be able to learn the relationship between the instances, and make a final decision based on the relationship. The transformer model is a good candidate for this task, as it is designed to learn the relationship between the tokens in the input sequence. In this blog, I will introduce how to use transformer model for multi-instance learning. The baseline model is the attention mechanism which is introduced in the [attension-based deep multiple instance learning (AD-MIL) model](https://arxiv.org/abs/1802.04712).

[Github repo](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning)

key takeaways:
1. The transformer model is a better modeling technique compared to AD-MIL model for multi-instance learning, especially when the task is more complex.
2. A ready-to-use transformer architecture is available in this repo as well as the training code for your custom dataset. The example dataloader for minst data is available. 

### Data set

The data set is the MNIST dataset, which is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. The dataset is available at [MNIST](http://yann.lecun.com/exdb/mnist/). The dataset is divided into 50,000 training images and 10,000 testing images. The images are 28x28 pixels, and the labels are 10 digits. The dataset is a good starting point for multi-instance learning, as the images are simple and the task is easy to understand.

![MNIST](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

To simulate multi-instance bag, we use a random distribution (Poisson or Normal) to generate the number of instances in each bag. The number of instances in each bag is capped by max_bag_length, and the mean bag length is specified by mnst_mean_bag_length. The random number of instances are selected randomly from the MNIST dataset, and the label depends on the task definition as described in the task design section.

### Multi-modal enbedding

We selected DINOv2 as our image encoder, and the DINO model is downloaded from the [Huggingface model hub](https://huggingface.co/docs/transformers/en/model_doc/dinov2). The DINO model is a vision transformer model that is pre-trained on the ImageNet dataset. The model weights are frozen and the image embedding from the DINO model (CLS TOKEN embedding) is used as the input to the multi-instalce learning model.

### Task design

The first task is a simple task, which is to determine if there is a target digit in the bag. The target digit can varies from 0 to 9. The label is 1 if the target digit is in the bag, and 0 otherwise. The task is a binary classification task, and the model should be able to learn the relationship between the instances in the bag and make a final decision based on the relationship.

![task1](/assets/MNST_bag_task1.png)

The second task is a slightly more complex task, which is to determine if there are multiple number of the target digit in the bag. For example, if the multiple number is 2, and target digit is 8, and there are at least 2 digit 8 in the bag, the label is 1, and 0 otherwise. The task is a binary classification task with a slightly more complex logic. 

![task2](/assets/MNST_bag_task2.png)


The last task is a complex task, which is to determine if there are two digits in the bag that sum up to a target number. For example, if the target number is 10, and there are two digits in the bag that sum up to 10, the label is 1, and 0 otherwise. The task is a binary classification task with a more complex logic.

![task3](/assets/MNST_bag_task3.png)

### Attention architectures 

For the multi-instance learning, we tested two attention machanism. The first attention architecture is a very light-weighted attention [MILAttention](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning/blob/main/tdmil/modelMIL.py#L13) introduced in the paper [attention-based deep multi-instance learning](https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py), and the second one is the [transformer based architecture](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning/blob/main/tdmil/modelMIL.py#L221) that is introduced in the paper [attention is all you need](https://arxiv.org/abs/1706.03762). We adaptived the architectures from the original code to fit the multi-instance learning task.


### Experimental Results

For the three tasks from simple to complex, we tested the two attention architectures. The results are shown in the following table. The transformer model is a better modeling technique compared to the MIL model for multi-instance learning, especially when the task is more complex. MIL model shows a performance drop as the task becomes more complex, while the transformer model shows a consistent performance across the tasks.

![fig1](/assets/MIL_fig1.png)


Another angle to look at the results is to compare the attention weights from the two models. The attention weights from the transformer model are the attention weights from the last block for the CLS token, and a mean aggregator was applied to the multiple head weights. The attention weights from the AD-MIL model are the straight forward attention score. 

For the simple task, both architectures show a similar attention pattern, where the attention weights are focused on the target digit. 

![fig2](/assets/MIL_Trans_attn_task1.png)

For the second task, both models shows a reasonable attention pattern, where the attention weights are focused on the target digit and the number of the target digit. However the MIL model fails to capture some blurry digits and leads to a wrong prediction. 

![fig3](/assets/Trans_attn_task2.png)
![fig4](/assets/MIL_attn_task2.png)


For the most complex task, the MIL model fails to capture the relationship between the digits, but only focus only on the digit 8. The transformer model shows a more complex attention pattern, where the attention weights are focused on the digits that sum up to the target number.

![fig5](/assets/Trans_attn_task3.png)
![fig6](/assets/MIL_attn_task3.png)


### Conclusion

We compared the transformer model with the MIL model for multi-instance learning. The transformer model is a better modeling technique compared to the MIL model for multi-instance learning, especially when the task is more complex. The transformer model shows a consistent performance across the tasks, while the MIL model shows a performance drop as the task becomes more complex. The attention weights from the transformer model are more interpretable compared to the MIL model, and the transformer model shows a more complex attention pattern compared to the MIL model.


### Repeat the experiments
In the [repo](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning) assiciated with this blog, a docker file is provided to manage the enviroment for this repo. You need to have GPU on your PC, and cuda 11.7+ and the GPU driver installed.

To open a jupyter notebook, run the following command:

```
## Open a jupyter notebook 
make env 
make jupyter 
```
To rerun the test, run the following command:
```
## Rerun a test
make env 
python 
python tdmil/train.py --config_file config/mnst_transformer_target_0.json
```

The configuration files are in the config folder, and you can change the configuration file to test different tasks and different attention architectures. The configuration files for the tests shown above are as follows, 

Task1: [MIL](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning/blob/main/config/mnst_mil_target_0.json), [Transformer](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning/blob/main/config/mnst_transformer_target_0.json) 
Task2: [MIL](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning/blob/main/config/mnst_mil_target_1.json), [Transformer](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning/blob/main/config/mnst_transformer_target_1.json) 
Task3: [MIL](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning/blob/main/config/mnst_mil_target_2.json), [Transformer](https://github.com/HoustonJ2013/transformer-deep-multi-instance-learning/blob/main/config/mnst_transformer_target_2.json) 
