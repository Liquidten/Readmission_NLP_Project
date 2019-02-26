import txt_preprocess
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, GlobalMaxPooling1D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard


if __name__ == "__main__":
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(cuis_preprocess.X_train)
    X_train_sequences = tokenizer.texts_to_sequences(cuis_preprocess.X_train)
    X_test_sequences = tokenizer.texts_to_sequences(cuis_preprocess.X_test)
    #X_val_sequences = tokenizer.texts_to_sequences(cuis_preprocess.X_val)
    word_index = tokenizer.word_index
    vocabulary_size = len(word_index)+1
    maxLen = len(max(X_train_sequences, key=len))
    X_train_sequences = sequence.pad_sequences(X_train_sequences, maxlen=maxLen)
    X_test_sequences = sequence.pad_sequences(X_test_sequences, maxlen=maxLen)
    #X_val_sequences = sequence.pad_sequences(X_val_sequences, maxlen=maxLen)

    # model
    model = Sequential()
    model.add(Embedding(vocabulary_size, 150, input_length=maxLen))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs/max2")
    x = model.fit(X_train_sequences,
                  cuis_preprocess.y_train,
                  validation_split=0.2,
                  batch_size=1,
                  epochs=20,
                  callbacks=[tensorboard])

    # train validation compare
    X_train_tmp, X_test_tmp, y_train_tmp, y_test_tmp = train_test_split(X_train_sequences,
                                                                        cuis_preprocess.y_train,
                                                                        test_size=0.2,
                                                                        random_state=1)
    train_pred = model.predict_proba(x=X_train_tmp,
                                     batch_size=1)
    fpr, tpr, _ = roc_curve(y_train_tmp, train_pred)
    print("max model training roc auc: " + str(auc(fpr, tpr)))
    val_pred = model.predict_proba(x=X_test_tmp,
                                   batch_size=1)
    fpr, tpr, _ = roc_curve(y_test_tmp, val_pred)
    print("max model validation roc auc: " + str(auc(fpr, tpr)))

    # graph
    class_pred = model.predict_classes(X_test_sequences, batch_size=16)
    acc = accuracy_score(cuis_preprocess.y_test, class_pred)
    print("deep max model test set accuracy " + str(acc))

    prob_pred = model.predict_proba(X_test_sequences, batch_size=16)
    roc_auc = roc_auc_score(cuis_preprocess.y_test, prob_pred) * 100
    print('{:0.2}'.format(roc_auc))

    loss = x.history['loss']
    val_loss = x.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', color='b', label='Training loss')
    plt.plot(epochs, val_loss, 'b', color='g', label='Validation loss')
    plt.title('Traning and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("/home/sshah33/Readmission/Graphs/max_1.png")

    plt.clf()
    acc = x.history['acc']
    val_acc = x.history['val_acc']
    plt.plot(epochs, acc, 'b', color='b', label='Training acc')
    plt.plot(epochs, val_acc, 'b', color='g', label='Validation acc')
    plt.title('Traning and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig("/home/sshah33/Readmission/Graphs/max_2.png")

    fpr, tpr, _ = roc_curve(cuis_preprocess.y_test, prob_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("/home/sshah33/Readmission/Graphs/max_3.png")
