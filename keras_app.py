from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from app.data import train_files, test_images
from app.prediction import generate_submission
from app.networks.keras_network import net


TRAIN_MODEL = True


def model():
    mod = net()
    mod.compile(
        loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy']
    )
    return mod


def train(x, y):
    model.fit(x, y, epochs=100, batch_size=128, verbose=1)
    model.save('models/keras/model.h5')


def evaluate(x, y):
    score = model.evaluate(x, y, 128)
    print('Eval Loss', score[0], 'Eval Accuracy:', score[1])


def load_trained_model():
    return load_model('models/keras/model.h5')


def predict(images):
    return model.predict(images)


if __name__ == '__main__':
    if TRAIN_MODEL:
        model = model()
        train_images, train_labels = train_files()
        train(train_images, train_labels)
    else:
        model = load_trained_model()

    eval_images = test_images()
    predictions = predict(eval_images)
    generate_submission(predictions)
