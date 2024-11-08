import marimo

__generated_with = "0.9.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import tensorflow as tf
    return (tf,)


@app.cell
def __(mo):
    import os
    import shutil
    from pathlib import Path

    for root, _, filenames in mo.status.progress_bar(list(os.walk("./dataset/"))):
        for filename in filenames:
            *_, font_name = root.split("/")
            file_path = os.path.join(root, filename)
            char_code = Path(filename).stem
            target_dir = os.path.join("dataset", char_code)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, f"{font_name}.png")
            shutil.copyfile(file_path, target_path)
    return (
        Path,
        char_code,
        file_path,
        filename,
        filenames,
        font_name,
        os,
        root,
        shutil,
        target_dir,
        target_path,
    )


@app.cell
def __(tf):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        "dataset/",
        validation_split=0.2,
        subset="training",
        seed=42,
        batch_size=32,
        image_size=(1024, 1024),
    )
    return (train_dataset,)


@app.cell
def __(tf):
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        "dataset/",
        validation_split=0.2,
        subset="validation",
        seed=42,
        batch_size=32,
        image_size=(1024, 1024),
    )
    return (val_dataset,)


@app.cell
def __(tf):
    @tf.function
    def add_alpha_channel(images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        alpha = tf.ones((batch_size, height, width, 1))
        return tf.concat([images, alpha], axis=-1)
    return (add_alpha_channel,)


@app.cell
def __(add_alpha_channel, tf):
    preprocessing = tf.keras.Sequential([
        tf.keras.layers.Resizing(
            1024,
            1024,
            pad_to_aspect_ratio=True,
        ),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Lambda(add_alpha_channel)  # Add alpha channel as the last step
    ])
    return (preprocessing,)


@app.cell
def __(preprocessing, train_dataset):
    aug_train_ds = train_dataset.map(lambda x, y: (preprocessing(x, training=True), y))
    return (aug_train_ds,)


@app.cell
def __(aug_train_ds):
    # Verify dataset shapes
    for images3, _ in aug_train_ds.take(1):
        print("Input shape:", images3.shape)
        print("Channels:", images3.shape[-1])
    return (images3,)


@app.cell
def __():
    input_width = 1024
    input_height = 1024
    channels = 4
    input_shape = (input_width, input_height, channels)
    latent_dim = 100
    return channels, input_height, input_shape, input_width, latent_dim


@app.cell
def __(aug_train_ds, tf):
    print("Checking shapes:")
    for images, _ in aug_train_ds.take(1):
        print("Training batch shape:", images.shape)
        print("Number of channels:", images.shape[-1])
        print("Value range:", tf.reduce_min(images).numpy(), "to", tf.reduce_max(images).numpy())
    return (images,)


@app.cell
def __(channels, latent_dim, tf):
    # Generator
    generator = tf.keras.Sequential([
        tf.keras.layers.Input((latent_dim,)),
        # Start with 8x8x256
        tf.keras.layers.Dense(8 * 8 * 256),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Reshape((8, 8, 256)),

        # 8x8 -> 16x16
        tf.keras.layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 16x16 -> 32x32
        tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 32x32 -> 64x64
        tf.keras.layers.Conv2DTranspose(32, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 64x64 -> 128x128
        tf.keras.layers.Conv2DTranspose(16, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 128x128 -> 256x256
        tf.keras.layers.Conv2DTranspose(8, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 256x256 -> 512x512
        tf.keras.layers.Conv2DTranspose(4, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 512x512 -> 1024x1024
        tf.keras.layers.Conv2DTranspose(channels, kernel_size=5, strides=2, padding="same", activation="sigmoid")
    ])

    # Discriminator
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024, 1024, 4)),

        # 1024x1024 -> 512x512
        tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 512x512 -> 256x256
        tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 256x256 -> 128x128
        tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 128x128 -> 64x64
        tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # 64x64 -> 32x32
        tf.keras.layers.Conv2D(512, kernel_size=5, strides=2, padding="same"),
        tf.keras.layers.PReLU(),

        # Final layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return discriminator, generator


@app.cell
def __(discriminator, generator, latent_dim, tf):
    print("Generator architecture:")
    generator.summary()

    print("\nDiscriminator architecture:")
    discriminator.summary()

    # Test generator output
    noise = tf.random.normal([1, latent_dim])
    generated = generator(noise, training=False)
    print("\nGenerator output shape:", generated.shape)
    return generated, noise


@app.cell
def __(tf):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return (cross_entropy,)


@app.cell
def __(cross_entropy, tf):
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    return (discriminator_loss,)


@app.cell
def __(cross_entropy, tf):
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    return (generator_loss,)


@app.cell
def __(tf):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    return discriminator_optimizer, generator_optimizer


@app.cell
def __(
    discriminator,
    discriminator_optimizer,
    generator,
    generator_optimizer,
    os,
    tf,
):
    checkpoint_dir = "./training_checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )
    return checkpoint, checkpoint_dir, checkpoint_prefix


@app.cell
def __(tf):
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    return EPOCHS, noise_dim, num_examples_to_generate, seed


@app.cell
def __(
    discriminator,
    discriminator_loss,
    discriminator_optimizer,
    generator,
    generator_loss,
    generator_optimizer,
    noise_dim,
    tf,
):
    BATCH_SIZE = 32

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return BATCH_SIZE, train_step


@app.cell
def __():
    import matplotlib.pyplot as plt

    def generate_and_save_images(model, epoch, test_input):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 * 127.5, cmap="gray")
            plt.axis("off")

        plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
        plt.show()
    return generate_and_save_images, plt


@app.cell
def __(
    EPOCHS,
    checkpoint,
    checkpoint_prefix,
    generate_and_save_images,
    generator,
    seed,
    train_step,
):
    import time


    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()

            for image_batch in dataset:
                train_step(image_batch)

            # display.clear_output(wait=True)

            generate_and_save_images(
                generator,
                epoch + 1,
                seed
            )

            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print("Time for epoch {} is {} sec.".format(epoch + 1, time.time() - start))


    # display.clear_output(wait=True)
    generate_and_save_images(
        generator,
        EPOCHS,
        seed
    )
    return time, train


@app.cell
def __(EPOCHS, aug_train_ds, train):
    train(aug_train_ds, EPOCHS)
    return


if __name__ == "__main__":
    app.run()
