def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
    print("Naive Bayes evaluation:\n{}".format('\n'.join(["OK: {} || test: {} || prediction: {}".format(
        str(t == l), str(t), str(l)) for t, l in zip(labels_test, pred)])))
    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    #ACCURACY BY HAND:
    classification_status = [a == b for a, b in zip(labels_test, pred)]
    accuracy = len([x for x in classification_status if x]) \
        / len(classification_status)
    #ACCURACY SCIKIT:
    accuracy = clf.score(features_test, labels_test)
    return accuracy
