# Predictive Coding based Incremental Learning in PyTorch

Predictive coding networks ([paper](https://arxiv.org/abs/2106.13082), [code](https://github.com/RobertRosenbaum/Torch2PC)) based implementation of Incremental Learning  ([paper](http://proceedings.mlr.press/v80/serra18a/serra18a.pdf), [code](https://github.com/joansj/hat)) in PyTorch. 

## Predictive Coding Algorithm

Inspired by the human brain, a predictive coding algorithm was introduced to resolve the biological limitation of backpropagation. Contrary to the neural plasticity of the human brain, the backpropagation algorithm performs global error-guided learning. However, in predictive coding, it performs local learning because its learning is performed with local error nodes in addition to the global error node. It has been demonstrated that an arbitrary computational graph can be trained in a predictive coding manner.

## Incremental Learning

The goal of incremental learning is acquiring new knowledge without forgetting previously learned information. This phenomenon is called catastrophic forgetting, which occurs when the network learns multiple tasks. However, the human can overcome that phenomenon using their synaptic plasticity. Preventing catastrophic forgetting is the key to incremental learning, and the current studies focus on the direction of finding these techniques.

## Training with Backpropagation

Please note that the training code is here just for demonstration purposes. 

To train the Protonet on this task, cd into this repo's `src` root folder and execute:

    $ python run.py
    
The script takes the following command line options:

- `root_save`: the root directory where tha checkpoint is stored, default to `'./checkpoint'`

- `experiment`: the type of task used for the experiment, default to `'mnist2'`

- `approach`: the learning algorithm used for the incremental learning, default to `'sgd'`

Running the command without arguments will train the models with the default hyperparamters values (producing results shown above).



## Training with Predictive Coding

Please note that the training code is here just for demonstration purposes. 

To train the predictive coding version of Protonet on this task with predictive coding manner, cd into this repo's `src` root folder and execute:

    $ python run_pc.py --error_type FixedPred --eta 0.1 --num_iter 20

- `error_type`: parameter update protocol of predictive coding algorithm, default to `FixedPred`

- `eta`: weight learning rate of predictive coding algorithm, default to `0.5`

- `num_iter`: the repetition number of backward iteration , default to `20`

The properties of other parameters are the same as backpropagation-based learning.


## .bib citation
cite the paper as follows (copied-pasted it from arxiv for you):
    
    @inproceedings{serra2018overcoming,
      title={Overcoming catastrophic forgetting with hard attention to the task},
      author={Serra, Joan and Suris, Didac and Miron, Marius and Karatzoglou, Alexandros},
      booktitle={International Conference on Machine Learning},
      pages={4548--4557},
      year={2018},
      organization={PMLR}
    }
        
    @article{rosenbaum2022relationship,
      title={On the relationship between predictive coding and backpropagation},
      author={Rosenbaum, Robert},
      journal={Plos one},
      volume={17},
      number={3},
      pages={e0266102},
      year={2022},
      publisher={Public Library of Science San Francisco, CA USA}
    }


## License

This project is licensed under the MIT License

Copyright (c) 2018 Daniele E. Ciriello, Orobix Srl (www.orobix.com).