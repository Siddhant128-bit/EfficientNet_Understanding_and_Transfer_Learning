<h3 align='center'> Efficient Net Study </h3>

<div align='left'> 
    <p> This Repo is about Efficient net, understanding what it is how it works and also how its archiecture and weights can be transfered to make it work on custom dataset
    </p>
    <p> The Dataset that we have used for ths particular project can be found here: </p>
        <ul>
            <li>https://www.kaggle.com/datasets/prasunroy/natural-images </li>
        </ul>
    </p>
    <p> We also have 2 local files that can be used to test the model's accuracy while inference testing </p>
    <ul>
        <li> test_1.jpg => Aeroplate </li>
        <li> test_2.jpg => Motorbike </li>
    </ul>
    <p> The Standard Architecture used for this project was that of Efficient Net B0 the complate architecture of the network can be seen as: </p>
    <img src='https://user-images.githubusercontent.com/80937266/222384110-0097b66e-ab94-4443-a90a-e68146a5f3af.png'>
</div>

<div>
    <p> We have initially loaded the model from kersa using the code work as: </p>
    
    from keras.applications.efficientnet import EfficientNetB0

    model=EfficientNetB0()
    
</div>

<div>      
    <p> Also for convinence and training we have freezed 75% of total layers and 25% of the layers have been pretrained ones.
    Model Architecture was built as shown below: </p>
    
         
    input_val=tf.keras.layers.Input(shape=(224,224,3))

    model=EfficientNetB0(include_top=False,input_tensor=input_val,weights='imagenet')

    for i in range(int(len(model.layers) * 0.75)):
        model.layers[i].trainable = False

    #model.trainable = False


    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = tf.keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = tf.keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    #x=tf.keras.layers.Flatten()
    output = tf.keras.layers.Dense(len(labels_map), activation="softmax", name="pred")(x)

    model=tf.keras.Model(inputs=input_val,outputs=output)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    
</div>

<div>
    <p> We have also used the early stopping as the call back that will monitor val loss and restore best weights during the loss between epochs with code as: </p>

    callback = tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss',mode='auto',restore_best_weights=True )

    hist=model.fit(X,Y,validation_data=(Xtest,Ytest),epochs=10,callbacks=[callback],batch_size=128)
</div>

<div>
    <p> The Graphs obtained from the training are of val_accuracy  and val_loss</p>
    <img src=https://user-images.githubusercontent.com/80937266/222404665-0ae31184-39c6-41d7-91d5-eb0501c2b4e8.png>
    <p>Also There might be an issue during model saving for that we recommend to downgrade tensorflow to verison 2.9 or less</p>
</div>

<div>
    <h4 align='center'> Inference Testing Examples: </h4>
    <img src=https://user-images.githubusercontent.com/80937266/222406317-4b7195c8-91ae-47e0-907a-c6ba56845d88.png>
</div>


