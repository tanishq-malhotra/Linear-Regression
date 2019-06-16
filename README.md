## Implementing Linear Regression from scratch in Python

Linear Regression is a technique in which a line which best fits our data is calculated.
On the basis of that line predictions are calculated.

## Linear Regression is based on the concept of:
![alt text][lr1]

[lr1]: https://cdn-images-1.medium.com/max/800/1*TEzHQl-H0E4YKOpInAN_ZQ.png

# or

![alt text][lr2]

[lr2]: https://cdn-images-1.medium.com/max/800/1*oNsv50WV-2gGXqpAT6Snag.png


where m is the slope, that is how steep is our line

## Finding the best fit line

1. m and b both have an initial value of 0.

2. We calculate the loss using these m and b value's.The loss used in Linear Regression is 
    Mean Squared Error which is given by the following equation:
    ![alt text][error]

    [error]: https://cdn-images-1.medium.com/max/1200/1*94Gc_tf4a5WPxxugOI5uWw.png

    Our Basic goal is to minimize this loss by finding an optimal m and b values.

3. Gradient Descent is a method used to find the optimal values of m and b by
    using the approach of partial derivative's.
    Gradient Descent:

    ![alt text][gd]

    [gd]: https://cdn-images-1.medium.com/max/1600/1*91DQMNKmNIdncqx6FsB4Iw.png


    We have to find the point in the local minima as depicted by the picture.
    Gradient Descent can be explained by the shape of a bowl.
    We have to go down gradiently untial point of local minima reached.
    This point of local minima will have the lowest error and goal is to minimize
    the error.

    Steps of GD:
    1. It draws the line(Tangent) from the point.
    2. It finds the slope of that line.
    3. It identifies how much change is required by taking the partial derivative of the function with respective to θ
    4. The change value will be multiplied with a variable called alpha(learning rate) we provide the value for alpha usually 0.01
    5. It subtracts this change value from the earlier θ value to get new θ value .

    The values of new m and b is calculated using bellow formula
    ![alt text][gdf]

    [gdf]: https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png


4. After the point of local minima is reached, we will get our optimal m and b values.
    Which can be now used to predict using the equation y=mx+b.


Initial value of Learning Rate is 0.01 and iterations are set to 1000.

You can also specify these values while creating the instance or object of the LinearRegression class.

A test file is also included with a dataset which shows how to use the model.

## These are the comparision between linear regression with sklearn and defined one:
![alt text][original]

original:- 

[original]: https://github.com/tanishq-malhotra/Linear-Regression-from-Scratch/blob/master/images/original.png?raw=true