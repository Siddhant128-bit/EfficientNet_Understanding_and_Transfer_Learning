<h3 align='center'> Efficient Net Study </h3>

<div align='left'> 
    <p> This Repo is about Efficient net, understanding what it is how it works and also how its archiecture and weights can be transfered to make it work on custom dataset
    </p>
    <p> The Dataset that we have used for ths particular project can be found here: </p>
        <ul>
            <li>https://www.kaggle.com/datasets/prasunroy/natural-images </li>
        </ul>
    </p>
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

