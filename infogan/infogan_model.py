import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from infogan.funcs import sample


class InfoGAN(Model):
    def __init__(self, generator, discriminator, recognition,
                 latent_spec, discrete_reg_coeff=1.0, continuous_reg_coeff=1.0):
        super(InfoGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.recognition = recognition

        self.latent_spec = latent_spec
        self.noise_var = latent_spec['noise-variables']
        self.cont_latent_dist = latent_spec['continuous-latent-codes'] # list of continuous latent codes
        self.disc_latent_dist = latent_spec['discrete-latent-codes'] # list of discrete latent codes
        self.latent_dist = self.cont_latent_dist + self.disc_latent_dist

        self.lambda_disc = discrete_reg_coeff
        self.lambda_cont = continuous_reg_coeff

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(InfoGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    @tf.function()
    def train_step(self, data):
        real_images = data
        batch_size = tf.shape(real_images)[0]
        noise_inputs, disc_inputs, cont_inputs = sample(self.latent_spec, batch_size)

        with tf.GradientTape(persistent=True) as gtape, tf.GradientTape(persistent=True) as dtape:

            fake_images = self.generator(noise_inputs)
            real_d, _   = self.discriminator(real_images)
            fake_d, mid = self.discriminator(fake_images)
            disc_outputs, cont_outputs = self.recognition(mid)

            dis_loss = d_loss = self.loss_fn(tf.ones_like(real_d), real_d) + self.loss_fn(tf.zeros_like(fake_d), fake_d)
            gen_loss = g_loss = self.loss_fn(tf.ones_like(fake_d), fake_d)
            discrete_loss, continuous_loss = 0, 0
            for cont_input, cont_output in zip(cont_inputs, cont_outputs):
                continuous_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_input-cont_output), -1)) * self.lambda_cont
                gen_loss += continuous_loss
            for disc_input, disc_output in zip(disc_inputs, disc_outputs):
                discrete_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(disc_input, disc_output) * self.lambda_disc
                gen_loss += discrete_loss

        g_vars = self.generator.trainable_weights + self.recognition.trainable_weights
        g_grads = gtape.gradient(gen_loss, g_vars)
        self.g_optimizer.apply_gradients(zip(g_grads, g_vars))

        d_vars = self.discriminator.trainable_weights
        d_grads = dtape.gradient(dis_loss, d_vars)
        self.d_optimizer.apply_gradients(zip(d_grads, d_vars))

        return {
            "G_loss": g_loss, "D_loss": d_loss,
            # "AA":  disc_mi_est ,
            # "D_Acc": self.dis_acc_tracker.result(),
            "MI": gen_loss - g_loss, # "Cross_Ent": cross_ent,
            "disc_loss": discrete_loss, "cont_loss": continuous_loss
            }

# class InfoGAN(Model):
#     def __init__(self, generator, discriminator, recognition, latent_spec):
#         super(InfoGAN, self).__init__()
#         self.generator = generator
#         self.discriminator = discriminator
#         self.recognition = recognition

#         self.latent_spec = latent_spec
#         self.noise_var = latent_spec['noise-variables']
#         self.cont_latent_dist = latent_spec['continuous-latent-codes'] # list of continuous latent codes
#         self.disc_latent_dist = latent_spec['discrete-latent-codes'] # list of discrete latent codes
#         self.latent_dist = self.cont_latent_dist + self.disc_latent_dist

#         self.info_reg_coeff = 1.0

#     def compile(self, g_optimizer, d_optimizer, loss_fn):
#         super(InfoGAN, self).compile()
#         self.g_optimizer = g_optimizer
#         self.d_optimizer = d_optimizer
#         self.loss_fn = loss_fn

#     @tf.function()
#     def train_step(self, data):
#         real_images = data
#         batch_size = tf.shape(data)[0]
#         noise_inputs, disc_inputs, cont_inputs = sample(self.latent_spec, batch_size)
#         # Train the discriminator
#         with tf.GradientTape(persistent=True) as dtape:
#             fake_images = self.generator(noise_inputs, training=False)
#             real_d, _   = self.discriminator(real_images)
#             fake_d, mid = self.discriminator(fake_images)
#             disc_outputs, cont_outputs = self.recognition(mid, training=False)

#             d_loss = self.loss_fn(tf.ones_like(real_d), real_d) + self.loss_fn(tf.zeros_like(fake_d), fake_d)
#             # cross_entropy(tf.ones_like(real_img), real_img) + cross_entropy(tf.zeros_like(fake_img), fake_img)

#             # g_loss = gen_loss(fake_d)
#             # info_loss = info(disc_inputs, disc_outputs, cont_inputs, cont_outputs)
#             #             gi = g_loss + info_loss
#             di = d_loss #+ info_loss

#         dvars = self.discriminator.trainable_weights
#         dgrads = dtape.gradient(d_loss, dvars)
#         self.d_optimizer.apply_gradients(zip(dgrads, dvars))

#         noise_inputs, disc_inputs, cont_inputs = sample(self.latent_spec, batch_size)
#         with tf.GradientTape(persistent=True) as gtape:
#             fake_images = self.generator(noise_inputs)
#             fake_d, mid = self.discriminator(fake_images, training=False)
#             disc_outputs, cont_outputs = self.recognition(mid)

#             # d_loss = dis_loss(real_d, fake_d)
#             g_loss = self.loss_fn(tf.ones_like(fake_d), fake_d)
#             c = 0
#             for cont_input, cont_output in zip(cont_inputs, cont_outputs):
#                 c += (tf.reduce_mean(tf.reduce_sum(tf.square(cont_input-cont_output), -1)) * 0.1)
#             sce = 0
#             for disc_input, disc_output in zip(disc_inputs, disc_outputs):
#                 sce += tf.keras.losses.CategoricalCrossentropy(from_logits=True)(disc_input, disc_output)
#             # c2 = tf.reduce_mean(tf.reduce_sum(tf.square(fkcon2-z_con2), -1)) * 0.5
#             # return c + sce
#             info_loss = c + sce #info(disc_inputs, disc_outputs, cont_inputs, cont_outputs)
#             gen_loss = g_loss + info_loss
#             # di = d_loss
#         g_vars = self.generator.trainable_weights + self.recognition.trainable_weights
#         g_grads = gtape.gradient(gen_loss, g_vars)
#         self.g_optimizer.apply_gradients(zip(g_grads, g_vars))

#         return {
#             "G_loss": g_loss, "D_loss": d_loss,
#             # "AA":  disc_mi_est ,
#             # "D_Acc": self.dis_acc_tracker.result(),
#             "MI": info_loss, # "Cross_Ent": cross_ent,
#         }