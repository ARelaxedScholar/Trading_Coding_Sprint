The core logic was written in the context of a coding sprint (a little over 3h.)
I then cleaned up the code, removing the .unwrap() that were written in for the sake
of testing the idea.

The idea was to implement a really simple trading idea, how much profit would we expect to make
assuming we were able to predict if a week from now, we'd make a profit.

Since it was a proof of concept, we only engineered features at a day level.
For QQQ using the free data from Investing.com and played with that.
The code is made such it would work with data taken from Investing.com.

    The goal is to create a AI signal tester. So we will need to train a model, and
    then use data to predict if entering right now would have generated a profit a week
    from now.

Currently for simplicity what we will do is always enter for exactly a week.
We will be using daily data.

We will take data and load it into a vector of Day structs.
Then we will run through the data to check if buying some action, say QQQ would generate profit.

We will be using a logistic regression as our model, nothing fancy.

General plan:

1. Load data

2. Do the feature engineering/analysis required in Rust.

3. Train the logistic regression on part of the data.

4. Run the algorithm on the rest of the data to see what would happen.
The goal is to see how much profit we would have made at the end.

Each part will be a function, and the main will really be calling those functions.
We won't be bothering with a GUI for now.
