'''
making sequential model in keras:
-simply, this is a linear layering of models
'''


#model = Sequential([list of models])
# or model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))



#needs to know input shape for first layer only
#pass in shape tuple in for input_shape=..., None allows any
inp = (784, )
model = Sequential()
model.add(Dense(32, input_shape=inp))
#can pass in batch_input_size instead
#or some layers allow input_dim (such as dense)


#merge layer allows combination of multiple layers



#when compliing layer, put optimizer, loss func to minimize, list of metrics 
model.compile(optimizer="adam", loss="categorical_cross_entropy", metrics=["accuracy"])


#train gwith fit
#pass in data, labels, num epochs, batch_size
model.fit(data, lables, num_epochs=2, batch_size=128)



#functional api for shared layers (eg. inception, etc.)
inputs = Input(shape=(784,)) #this returns a tensor
#create 3 layer model with predictions at end
model = Model(input=inputs, output=predictions)



#can resuse models on other data by simply calling it on that code
x = Input(shape=(784,))
y = model(x)
#can use existing models on new tasks
from keras.layers import TimeDistributed
input_sequences = Input(shape=(20, 784))
processed_sequences = TimeDistributed(model)(input_sequences)




#can have multiple inputs as well as multiple outputs
model = Model(input=[main, auxillary], output=[main_output, auxillary_output])
#can create inception module by merging multiple towers with mode=concat




#saving models ==
model.save(filepath)

# use it (and recompile automatically)
keras.models.load_model(filepath)

#early stopping function when validation loss isn't decreassing
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
callbacks=[early_stopping]

#can print history of model to see weights #can set layer to not be trainable (frozen)


'''
generate additional images 
'''
datagen = ImageDataGenerator(
  horizontal_flip=True)

img = load_img('test2.png')
x = img_to_array(img)
x = x.reshape((1, ) + x.shape)
# print('x shape before', x.shape)
# x = np.moveaxis(x, 3, 1)
print('x shape after', x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 1, save_to_dir='preview', save_prefix='new_test', save_format='png'):
  print('running with i', i)
  i += 1
  if i > 20:
    break
