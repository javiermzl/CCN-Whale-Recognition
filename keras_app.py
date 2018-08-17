from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from app.data import generate_train_files, generate_eval_images
from app.prediction import generate_submission
from app.networks.keras_network import net


def train():
    model = net()
    model.compile(
        loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy']
    )

    print('Importing Data')
    images, labels = generate_train_files()

    model.fit(images, labels, epochs=150, batch_size=100, verbose=1)
    model.save('models/keras/whale_model.h5')

    return model


def recover_model():
    return load_model('models/keras/whale_model.h5')


def predict(model):
    eval_x = generate_eval_images()
    return model.predict(eval_x)


if __name__ == '__main__':
    mod = train()
    # mod = recover_model()
    predictions = predict(mod)
    generate_submission(predictions)
