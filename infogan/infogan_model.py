import tensorflow as tf
from infogan.utils import sample
from infogan.distributions import GaussianNLLLoss, LogProb


class InfoGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator, recognition,
                 latent_spec, discrete_reg_coeff=1.0, continuous_reg_coeff=1.0):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.recognition = recognition

        self.latent_spec = latent_spec
        # self.noise_var = latent_spec['noise-variables']
        # self.cont_latent_dist = latent_spec['continuous-latent-codes']  # list of continuous latent codes
        # self.disc_latent_dist = latent_spec['discrete-latent-codes']    # list of discrete latent codes

        self.lambda_disc = discrete_reg_coeff
        self.lambda_cont = continuous_reg_coeff
        self.continuous_loss = GaussianNLLLoss()
        self.log_prob = LogProb()
        self.g_optimizer, self.d_optimizer = None, None
        self.loss_fn = None

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

            disc_loss, cont_loss = 0, 0
            for cont_input, cont_output in zip(cont_inputs, cont_outputs):
                z, z_mean, z_log_var = cont_output
                cont_cross_ent = self.continuous_loss(cont_input, z_mean, z_log_var)
                cont_mi_est = - cont_cross_ent  # ignore H(c) in L_I
                gen_loss -= self.lambda_cont * cont_mi_est
                # dis_loss -= self.lambda_cont * cont_mi_est
                cont_loss += cont_cross_ent

            for disc_input, disc_output in zip(disc_inputs, disc_outputs):
                disc_cross_ent = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(disc_input, disc_output)
                disc_mi_est = - disc_cross_ent  # H(c) is constant so ignore it in MI = H(c) - H(c|x)
                gen_loss -= self.lambda_disc * disc_mi_est
                # dis_loss -= self.lambda_disc * disc_mi_est
                disc_loss += disc_cross_ent

        g_vars = self.generator.trainable_weights + self.recognition.trainable_weights
        g_grads = gtape.gradient(gen_loss, g_vars)
        self.g_optimizer.apply_gradients(zip(g_grads, g_vars))

        d_vars = self.discriminator.trainable_weights
        d_grads = dtape.gradient(dis_loss, d_vars)
        self.d_optimizer.apply_gradients(zip(d_grads, d_vars))

        return {"G_loss": g_loss, "D_loss": d_loss, "Info_loss": dis_loss+cont_loss,
                "Disc_loss": disc_loss, "Cont_loss": cont_loss}


