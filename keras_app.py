from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from app.prediction import generate_submission
from app.networks.keras_network import net
from app import data


def train():
    model = net()
    adam = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print('Importing Data')
    images, labels = data.gen_raw_data()

    history = model.fit(images, labels, epochs=150, batch_size=100, verbose=1)
    model.save('models/keras/whale_model.h5')
    return model


def recover_model():
    return load_model('models/keras/whale_model.h5')


def predict(model):
    eval_x, eval_labels = data.import_eval_files()
    return model.predict(eval_x)


# mod = train()
mod = recover_model()
predictions = predict(mod)
generate_submission(predictions)
