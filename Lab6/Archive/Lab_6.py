import numpy as np

np.random.seed(1)

#Problem 1

#Problem 1.1
def gini_score(groups, classes):
    '''
    Inputs: 
    groups: 2 lists of examples. Each example is a list, where the last element is the label.
    classes: a list of different class labels (it's simply [0.0, 1.0] in this problem)
    Outputs:
    gini: gini score, a real number
    '''
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    group1 = groups[0]
    group2 = groups[1]

    # calculate n samples in groups
    g1_n = len(group1)
    g2_n = len(group2)

    # initialize g1, g2
    g1, g2 = 0, 0

    # number of samples of class 0
    g1_class_0 = 0

    # only calculate gini if g1_n != 0
    if g1_n != 0:
      for i in group1:
        last_element = i[-1]
        g1_class_0 += i.count(classes[0])
    
      # number of samples of class 1
      g1_class_1 = g1_n - g1_class_0

      # calculate g1
      g1 = 1 - ((g1_class_1/g1_n)**2 + (g1_class_0/g1_n)**2)
    
    # number of samples of class 0
    g2_class_0 = 0
  
    # only calculate gini if g2_n != 0
    if g2_n != 0:
      for i in group2:
        last_element = i[-1]
        g2_class_0 += i.count(classes[0])
  
      # number of samples of class 1
      g2_class_1 = g2_n - g2_class_0

      # calculate g2
      g2 = 1 - ((g2_class_1/g2_n)**2 + (g2_class_0/g2_n)**2)

    # calculate gini
    # n samples in group 1 + group 2
    n = g1_n + g2_n

    # calculate gini
    gini = (g1_n/n)*g1 + (g2_n/n)*g2
 
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return gini

#Problem 1.2
def create_split(index, threshold, datalist):
    '''
    Inputs:
    index: The index of the feature used to split data. It starts from 0.
    threshold: The threshold for the given feature based on which to split the data.
        If an example's feature value is < threshold, then it goes to the left group.
        Otherwise (>= threshold), it goes to the right group.
    datalist: A list of samples. 
    Outputs:
    left: List of samples
    right: List of samples
    '''
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # initiate left and right lists
    left = []
    right = []
    # iterate over samples of feature = threshold
    for list in datalist:
      # if < threshold send to left group
      if list[index] < threshold:
        left.append(list)

      # if >= send to right
      else: 
        right.append(list)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return left, right

#Problem 1.3
def get_best_split(datalist):
    '''
    Inputs:
    datalist: A list of samples. Each sample is a list, the last element is the label.
    Outputs:
    node: A dictionary contains 3 key value pairs, such as: node = {'index': integer, 'value': float, 'groups': a tuple contains two lists of examples}
    Pseudo-code:
    for index in range(#feature): # index is the feature index
    for example in datalist:
        use create_split with (index, example[index]) to divide datalist into two groups
        compute the Gini index for this division
    construct a node with the (index, example[index], groups) that corresponds to the lowest Gini index
    '''
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # intialize node
    node = {}

    # Initialize minimum Gini index to 1
    min_gini = 1
    classes = []

    # start loop every feature and every possible value of the feature in the datalist
    for index in range(len(datalist[0])-1): # index is the feature index
    
      for example in datalist:
        # find class values
        if example[-1] not in classes:
          classes.append(example[-1])

        #split datalist
        split_data = create_split(index, example[index], datalist)
    
        # calculate gini score
        gini = gini_score(split_data, classes)
    
        # check gini score with min_gini
        if gini < min_gini:
          min_gini = gini
          #construct a node with the (index, example[index], groups) that corresponds to the lowest Gini index
          node = {'index': index, 'value': example[index], 'groups': split_data}
  

  
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return node

#Problem 1.4
#Problem 1.4-step 1
def to_terminal(group):
    '''
    Input:
    group: A list of examples. Each example is a list, whose last element is the label.
    Output:
    label: the label indicating the most common class value in the group
    '''
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    classes = {}

    # loop through examples, using dictionary to capture unique class and freq. count
    for example in group: 
      if (example[-1] in classes):
          classes[example[-1]] += 1
      else:
          classes[example[-1]] = 1
    # label is the dict key with max value
    label = max(classes, key=lambda k: classes[k])
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return label

#Problem 1.4-step 2
def recursive_split(node, max_depth, min_size, depth):
    '''
    Inputs:
    node:  A dictionary contains 3 key value pairs, node = 
            {'index': integer, 'value': float, 'groups': a tuple contains two lists fo samples}
    max_depth: maximum depth of the tree, an integer
    min_size: minimum size of a group, an integer
    depth: tree depth for current node
    Output:
    node: as defined above
    '''
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # extract left and right groups from input node
    left_g, right_g = node['groups']
    
    # delete groups in node to save space
    del node['groups']
  
    # check if left or right node is empty, set label for both to max class on non-empty node 
    if len(left_g) == 0:
      node['left'] = to_terminal(right_g)
      node['right'] = to_terminal(right_g)
    elif len(right_g) == 0:
      node['left'] = to_terminal(left_g)
      node['right'] = to_terminal(left_g)
    else: 

      # check tree depth against max depth  
      if depth >= (max_depth -1):
        # add termination on left and right if depth = max depth
        node['left'] = to_terminal(left_g)
        node['right'] = to_terminal(right_g)
      else:
        
        # process left child
        if len(left_g) <= min_size:
          node['left'] = to_terminal(left_g)
        else:
          node['left'] = get_best_split(left_g)
          recursive_split(node['left'], max_depth, min_size, depth+1)
       
        # process right child
        if len(right_g) <= min_size:
          node['right'] = to_terminal(right_g)
        else:
          node['right'] = get_best_split(right_g)
          recursive_split(node['right'], max_depth, min_size, depth+1)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return node

#Problem 1.4-step 3
def build_tree(train, max_depth, min_size):
    '''
    Inputs:
    - train: Training set, a list of examples. Each example is a list, whose last element is the label.
    - max_depth: maximum depth of the tree, an integer (root has depth 1)
    - min_size: minimum size of a group, an integer
    Output:
    - root: The root node, a recursive dictionary that should carry the whole tree
    '''
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    root = get_best_split(train)
    recursive_split(root, max_depth, min_size, depth = 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return root

#Problem 1.4-step 4
def predict(root, sample):
    '''
    Inputs:
    root: the root node of the tree. a recursive dictionary that carries the whole tree.
    sample: a list
    Outputs:
    '''
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    node = root

    # continue through the tree until terminal node reached
    while isinstance(node, dict):
      if sample[node['index']] < node['value']:
        node = node['left']
      else:
        node = node['right']

    # Return the label of terminal node
    return node
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#Problem 2
def Problem2():
    '''
    Inputs: 
    Outputs:
    acc: accuracy score, a real number
    f1: f1 score, a real number
    '''
    # load and prepare data
    import pandas as pd
    import urllib.request
    import shutil
    from csv import reader
    from random import seed

    url = 'https://www.cs.uic.edu/~zhangx/teaching/data_banknote_authentication.csv'
    file_name = 'data_banknote_authentication.csv'
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    file = open(file_name, "rt")
    lines = reader(file)

    df = pd.read_csv(file_name,
                        sep=',',
                        header=None)
    df.head()

    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    seed(1)

    dataset = list(lines)
    max_depth = 6
    min_size = 10
    num_train = 1000

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # convert to float
    df = df.apply(pd.to_numeric)

    #split dataset
    train = df.iloc[:1000]
    test = df.iloc[1000:]

    # build tree
    root = build_tree(train.values.tolist(), max_depth, min_size)

    # make prediction
    predictions = [predict(root, sample) for sample in test.values.tolist()]

    #print accuracy and f1
    acc = accuracy_score(test.iloc[:, -1].tolist(), predictions)
    f1 = f1_score(test.iloc[:, -1].tolist(), predictions)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc, f1
