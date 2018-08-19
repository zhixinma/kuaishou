# coding: utf-8
import mxnet as mx
from mxnet import nd
import numpy as np
from mxnet.gluon import nn, rnn
from mxnet import autograd

class ncf_it(nn.Block):
    def __init__(self, nb_users, nb_photos, vocabulary_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_it, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_user  = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_user = nn.Dense(units=word_dim, activation='relu')
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')
            self.mlp_1 = nn.Dense(units=128, activation='relu')
            self.mlp_2 = nn.Dense(units=64, activation='relu')
            self.mlp_3 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, photo_id, user_doc, text):
        user_emb = self.emb_user(user_id)
        text_emb = self.emb_word(text)
        text_emb = self.mlp_word(text_emb)
        user_doc_emb = self.emb_user(user_doc)
        user_doc_emb = self.mlp_user(user_doc_emb)
        x = nd.concat(user_emb, user_doc_emb, text_emb, dim=1)
        x_1 = self.mlp_1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.mlp_2(x_1)
        res = self.mlp_3(x_2)
        return res


class ncf_it_interaction(nn.Block):
    def __init__(self, nb_users, nb_photos, vocabulary_size, batch_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_it_interaction, self).__init__(**kwargs)
        with self.name_scope():
            self.batch_size = batch_size
            self.emb_user  = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_user = nn.Dense(units=word_dim, activation='relu')
            self.mlp_1 = nn.Dense(units=128, activation='relu')
            self.mlp_2 = nn.Dense(units=64, activation='relu')
            self.mlp_3 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, photo_id, user_doc, text):
        user_emb = self.emb_user(user_id)
        text_emb = self.emb_word(text)
        # 
        # print("user_emb", user_emb.shape)
        # print("text_emb", text_emb.shape)
        text_emb = nd.transpose(text_emb,(0,2,1))
        # print("text_emb", text_emb.shape)
        user_emb_i = user_emb.reshape((self.batch_size,1,-1))
        # print("user_emb_i", user_emb_i.shape)
        interaction = nd.batch_dot(user_emb_i, text_emb)
        interaction = interaction.reshape((self.batch_size, -1))
        # print("interaction", interaction.shape)
        # 
        user_doc_emb = self.emb_user(user_doc)
        user_doc_emb = self.mlp_user(user_doc_emb)
        x = nd.concat(user_emb, user_doc_emb, interaction, dim=1)
        x_1 = self.mlp_1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.mlp_2(x_1)
        res = self.mlp_3(x_2)
        return res

# 2048 降维 作为特征
class ncf_uit(nn.Block):
    def __init__(self, nb_users, nb_photos, vocabulary_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_uit, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_user  = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_user = nn.Dense(units=word_dim, activation='relu')
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')

            self.mlp_v1 = nn.Dense(units=1024, activation='relu')
            self.mlp_v2 = nn.Dense(units=512, activation='relu')
            self.mlp_v3 = nn.Dense(units=128, activation='relu')

            self.mlp_1 = nn.Dense(units=128, activation='relu')
            self.mlp_2 = nn.Dense(units=64, activation='relu')
            self.mlp_3 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, userep, photo_id, user_doc, text):
        user_emb = self.emb_user(user_id)
        text_emb = self.emb_word(text)
        text_emb = self.mlp_word(text_emb)

        user_doc_emb = self.emb_user(user_doc)
        user_doc_emb = self.mlp_user(user_doc_emb)

        userep_emb = self.mlp_v1(userep)
        userep_emb = self.mlp_v2(userep_emb)
        userep_emb = self.mlp_v3(userep_emb)

        x = nd.concat(user_emb, user_doc_emb, text_emb, userep_emb, dim=1)

        x_1 = self.mlp_1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.mlp_2(x_1)
        res = self.mlp_3(x_2)
        return res

class ncf_uit_2048(nn.Block):
    def __init__(self, nb_users, nb_photos, vocabulary_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_uit_2048, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_user  = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_user = nn.Dense(units=word_dim, activation='relu')
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')

            # self.mlp_v1 = nn.Dense(units=1024, activation='relu')
            # self.mlp_v2 = nn.Dense(units=512, activation='relu')
            # self.mlp_v3 = nn.Dense(units=128, activation='relu')

            self.mlp_1 = nn.Dense(units=1024, activation='relu')
            self.mlp_2 = nn.Dense(units=512, activation='relu')
            self.mlp_3 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, userep, photo_id, user_doc, text):
        user_emb = self.emb_user(user_id)
        text_emb = self.emb_word(text)
        text_emb = self.mlp_word(text_emb)

        user_doc_emb = self.emb_user(user_doc)
        user_doc_emb = self.mlp_user(user_doc_emb)

        # userep_emb = self.mlp_v1(userep)
        # userep_emb = self.mlp_v2(userep_emb)
        # userep_emb = self.mlp_v3(userep_emb)

        x = nd.concat(user_emb, user_doc_emb, text_emb, userep, dim=1)

        x_1 = self.mlp_1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.mlp_2(x_1)
        res = self.mlp_3(x_2)
        return res


class ncf_ueit(nn.Block):
    def __init__(self, nb_users, nb_photos, vocabulary_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_ueit, self).__init__(**kwargs)
        with self.name_scope():
            # initialized with pretrained data
            # user_doc and sentence dimensions are 512 
            self.emb_user  = nn.Embedding(nb_users, 2048)
            self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_user = nn.Dense(units=512, activation='relu')
            self.mlp_word = nn.Dense(units=512, activation='relu')
            # self.mlp_v1 = nn.Dense(units=1024, activation='relu')
            # self.mlp_v2 = nn.Dense(units=512, activation='relu')
            # self.mlp_v3 = nn.Dense(units=128, activation='relu')
            self.mlp_1 = nn.Dense(units=1024, activation='relu')
            self.mlp_2 = nn.Dense(units=512, activation='relu')
            self.mlp_3 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, userep, photo_id, user_doc, text):
        user_emb = self.emb_user(user_id)
        text_emb = self.emb_word(text)
        text_emb = self.mlp_word(text_emb)
        user_doc_emb = self.emb_user(user_doc)
        user_doc_emb = self.mlp_user(user_doc_emb)
        # userep_emb = self.mlp_v1(userep)
        # userep_emb = self.mlp_v2(userep_emb)
        # userep_emb = self.mlp_v3(userep_emb)
        x = nd.concat(user_emb, user_doc_emb, text_emb, dim=1)
        x_1 = self.mlp_1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.mlp_2(x_1)
        res = self.mlp_3(x_2)
        return res


class ncf_it_asc(nn.Block):
    # sentence dimension = 1024
    def __init__(self, nb_users, nb_photos, vocabulary_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_it_asc, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_user  = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_user = nn.Dense(units=word_dim, activation='relu')
            self.mlp_word = nn.Dense(units=1024, activation='relu')
            self.mlp_1 = nn.Dense(units=512, activation='relu')
            self.mlp_2 = nn.Dense(units=128, activation='relu')
            self.mlp_3 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, photo_id, user_doc, text):
        user_emb = self.emb_user(user_id)
        text_emb = self.emb_word(text)
        text_emb = self.mlp_word(text_emb)
        user_doc_emb = self.emb_user(user_doc)
        user_doc_emb = self.mlp_user(user_doc_emb)
        x = nd.concat(user_emb, user_doc_emb, text_emb, dim=1)
        x_1 = self.mlp_1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.mlp_2(x_1)
        res = self.mlp_3(x_2)
        return res

class ncf_it_cross(nn.Block):
    def __init__(self, nb_users, nb_photos, vocabulary_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_it_cross, self).__init__(**kwargs)
        with self.name_scope():
            length = 4424
            self.emb_user  = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_user = nn.Dense(units=word_dim, activation='relu')
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')
            self.flatten = nn.flatten()

            self.fc_0 = nn.Dense(units=1024)
            self.fc_1 = nn.Dense(units=1024)
            self.fc_2 = nn.Dense(units=1024)
            self.fc_3 = nn.Dense(units=1024)

            self.w0 = nd.random.uniform(0, 1, (1, length))
            self.w1 = nd.random.uniform(0, 1, (1, length))
            self.w2 = nd.random.uniform(0, 1, (1, length))
            self.w3 = nd.random.uniform(0, 1, (1, length))
            self.wl = nd.random.uniform(0, 1, (1, length))

            self.b0 = nd.random.uniform(-1, 1, (1, length))
            self.b1 = nd.random.uniform(-1, 1, (1, length))
            self.b2 = nd.random.uniform(-1, 1, (1, length))
            self.b3 = nd.random.uniform(-1, 1, (1, length))
            self.bl = nd.random.uniform(-1, 1, (1, length))


            self.mlp_1 = nn.Dense(units=1024, activation='relu')
            self.mlp_2 = nn.Dense(units=258, activation='relu')
            # self.mlp_3 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, photo_id, user_doc, text):
        ############## Embedding ###############
        user_emb = self.emb_user(user_id)
        text_emb = self.emb_word(text)
        user_doc_emb = self.emb_user(user_doc)
        ################ Cross #################
        user_0 = self.flatten(user_emb)
        text_0 = self.flatten(text_emb)
        udoc_0 = self.flatten(user_doc_emb)
        x_c0 = nd.concat(user_0, text_0, udoc_0)

        x_c1 = x_c0*x_c0*w0 + b0 + x_c0
        x_c2 = x_c0*x_c1*w1 + b1 + x_c1
        x_c3 = x_c0*x_c2*w2 + b2 + x_c2
        x_c4 = x_c0*x_c3*w3 + b3 + x_c3

        ################# DNN ##################
        text_emb = self.mlp_word(text_emb)
        user_doc_emb = self.mlp_user(user_doc_emb)
        x_d0 = nd.concat(user_emb, user_doc_emb, text_emb, dim=1)
        x_d1 = self.mlp_1(x_d0)
        x_d1 = self.dropout(x_d1)
        x_d2 = self.mlp_2(x_d1)
        ################# Stack ################

        res = self.mlp_3(x_d2)
        return res

class rnn_net(nn.Block):
    def __init__(self, 
                 mode,
                 vocab_size,
                 num_embed,
                 num_hidden,
                 num_layers,
                 dropout_rnn=0.25,
                 dropout=0.25,
                 **kwargs):
        super(rnn_net, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout_rnn)
            self.mlp = nn.Dense(units=num_hidden, activation='relu')
            self.encoder = nn.Embedding(vocab_size, num_embed, weight_initializer=mx.init.Uniform(0.1))
            if mode == 'lstm':
                self._rnn = rnn.LSTM(num_hidden, num_layers, layout='NTC', dropout=dropout_rnn, input_size=num_embed)
            elif mode == 'gru':
                self._rnn = rnn.GRU(num_hidden, num_layers, layout='NTC', dropout=dropout_rnn, input_size=num_embed)
            else:
                self._rnn = rnn.RNN(num_hidden, num_layers, layout='NTC', activation='relu', dropout=dropout_rnn, input_size=num_embed)
            self.num_hidden = num_hidden
    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        output, hidden = self._rnn(emb, hidden)
        output = self.drop(output)
        output = self.mlp(output)
        return output, hidden

class cnn(nn.Block):
    def __init__(self, **kwargs):
        super(cnn, self).__init__(**kwargs)
        with self.name_scope():
            self.pooling_1 = nn.MaxPool2D(pool_size=8, strides=8)
            self.pooling_2 = nn.MaxPool2D(pool_size=8, strides=8)
            self.pooling_3 = nn.MaxPool2D(pool_size=8, strides=8)
            self.conv1 = nn.Conv2D(channels=64,kernel_size=3)
            self.conv2 = nn.Conv2D(channels=64,kernel_size=3)
            self.conv3 = nn.Conv2D(channels=64,kernel_size=3)
            self.mlp = nn.Dense(units=1024,activation='tanh',flatten=True)
    def forward(self, data):
        x_1 = self.conv1(data)
        x_1 = self.pooling_1(x_1)
        x_2 = self.conv2(x_1)
        x_2 = self.pooling_2(x_2)
        x_3 = self.conv3(x_2)
        x_3 = self.pooling_3(x_3)
        res = self.mlp(x_3)
        return res

class ncf_v(nn.Block):
    def __init__(self, nb_users, word_dim=128, visual_dim=2048, visual_emb=512, vocabulary_size=233243, **kwargs):
        super(ncf_v, self).__init__(**kwargs)
        with self.name_scope():
            self.dense_v = nn.Dense(units=visual_emb, activation='relu')
            self.emb_uv = nn.Embedding(nb_users, visual_emb, weight_initializer=mx.init.Uniform(1))
            self.mlp_v1 = nn.Dense(units=256, activation='relu')
            self.mlp_v2 = nn.Dense(units=64, activation='relu')
            self.mlp = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, text, visual):
        user_vis = self.emb_uv(user_id)
        visual_emb = self.dense_v(visual)
        xv = nd.concat(user_vis, visual_emb, dim=1)
        xv_1 = self.mlp_v1(xv)
        xv_2 = self.mlp_v2(xv_1)
        res = self.mlp(xv_2)
        return res

class ncf_vt(nn.Block):
    def __init__(self, nb_users, word_dim=128, visual_dim=2048, visual_emb=512, vocabulary_size=233243, **kwargs):
        super(ncf_vt, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_uw = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dense_v = nn.Dense(units=visual_emb, activation='relu')
            self.emb_uv = nn.Embedding(nb_users, visual_emb, weight_initializer=mx.init.Uniform(1))
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')
            self.mlp_w1 = nn.Dense(units=128, activation='relu')
            self.mlp_w2 = nn.Dense(units=64, activation='relu')
            self.mlp_v1 = nn.Dense(units=256, activation='relu')
            self.mlp_v2 = nn.Dense(units=64, activation='relu')
            self.mlp = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, text, visual):
        user_word = self.emb_uw(user_id)
        word_emb = self.emb_word(text)
        word_emb = self.mlp_word(word_emb)
        user_vis = self.emb_uv(user_id)
        visual_emb = self.dense_v(visual)
        xv = nd.concat(user_vis, visual_emb, dim=1)
        xw = nd.concat(user_word, word_emb, dim=1)
        xv_1 = self.mlp_v1(xv)
        xv_2 = self.mlp_v2(xv_1)
        xw_1 = self.mlp_w1(xw)
        xw_2 = self.mlp_w2(xw_1)
        x = nd.concat(xw_2, xv_2, dim=1)
        res = self.mlp(x)
        return res

class ncf_vtf(nn.Block):
    def __init__(self, nb_users, word_dim=128, visual_dim=2048, visual_emb=512, vocabulary_size=233243, **kwargs):
        super(ncf_vtf, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_uw = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dense_v = nn.Dense(units=visual_emb, activation='relu')
            self.emb_uv = nn.Embedding(nb_users, visual_emb, weight_initializer=mx.init.Uniform(1))
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')
            self.mlp_w1 = nn.Dense(units=128, activation='relu')
            self.mlp_w2 = nn.Dense(units=64, activation='relu')
            self.mlp_v1 = nn.Dense(units=256, activation='relu')
            self.mlp_v2 = nn.Dense(units=64, activation='relu')
            self.mlp_1 = nn.Dense(units=4, activation='relu')
            self.mlp_2 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, text, visual, face):
        user_word = self.emb_uw(user_id)
        word_emb = self.emb_word(text)
        word_emb = self.mlp_word(word_emb)
        user_vis = self.emb_uv(user_id)
        visual_emb = self.dense_v(visual)
        xv = nd.concat(user_vis, visual_emb, dim=1)
        xw = nd.concat(user_word, word_emb, dim=1)
        xv_1 = self.mlp_v1(xv)
        xv_2 = self.mlp_v2(xv_1)
        xw_1 = self.mlp_w1(xw)
        xw_2 = self.mlp_w2(xw_1)
        x = nd.concat(xw_2, xv_2, dim=1)
        x_1 = self.mlp_1(x)
        x_2 = nd.concat(x_1, face, dim=1)
        res = self.mlp_2(x_2)
        return res

class ncf_f(nn.Block):
    def __init__(self, nb_users, age_size, score_size, word_dim=128, visual_dim=2048, visual_emb=512, vocabulary_size=233243, **kwargs):
        super(ncf_f, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_user = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_gender = nn.Embedding(3, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_age = nn.Embedding(age_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_score = nn.Embedding(score_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.mlp_1 = nn.Dense(units=128, activation='relu')
            self.mlp_2 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, face_batch):
        user_emb = self.emb_user(user_id)
        gender_emb = self.emb_gender(face_batch[:,0])
        age_emb = self.emb_age(face_batch[:,1])
        score_emb = self.emb_score(face_batch[:,2])
        x = nd.concat(user_emb, gender_emb, age_emb, score_emb, dim=1)
        x_1 = self.mlp_1(x)
        res = self.mlp_2(x_1)
        return res

class ncf_vf(nn.Block):
    def __init__(self, nb_users, age_size, score_size, word_dim=128, visual_dim=2048, visual_emb=512, vocabulary_size=233243, **kwargs):
        super(ncf_vf, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_user = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_gender = nn.Embedding(3, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_age = nn.Embedding(age_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_score = nn.Embedding(score_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.mlp_v1 = nn.Dense(units=256, activation='relu')
            self.mlp_1 = nn.Dense(units=128, activation='relu')
            self.mlp_2 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, face_batch, visual):
        user_emb = self.emb_user(user_id)
        gender_emb = self.emb_gender(face_batch[:,0])
        age_emb = self.emb_age(face_batch[:,1])
        score_emb = self.emb_score(face_batch[:,2])
        visual_emb = self.mlp_v1(visual)
        x = nd.concat(user_emb, gender_emb, age_emb, score_emb, visual_emb, dim=1)
        x_1 = self.mlp_1(x)
        res = self.mlp_2(x_1)
        return res

class ncf_topic(nn.Block):
    def __init__(self, nb_users, topics_num, sentence_length, batch_size=1178, word_dim=128, visual_dim=2048, visual_emb=512, vocabulary_size=233243, **kwargs):
        super(ncf_topic, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_uw = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.mlp_topic = nn.Dense(units=word_dim, activation='relu')
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')
            self.mlp_w1 = nn.Dense(units=128, activation='relu')
            self.mlp_w2 = nn.Dense(units=64, activation='relu')
            self.mlp = nn.Dense(units=1, activation='sigmoid')
            self.topics_num = topics_num
            self.word_dim = word_dim
            self.batch_size = batch_size
            self.sentence_length = sentence_length
    def forward(self, user_id, text, topics):
        user_word = self.emb_uw(user_id)
        word_emb = self.emb_word(text)
        topics_emb = self.emb_word(topics)
        topics_emb = nd.transpose(topics_emb, axes=(1,0))
        topics_emb = nd.reshape(topics_emb, (self.word_dim,self.topics_num,1))
        topics_emb = nd.dot(word_emb, topics_emb)
        topics_emb = nd.reshape(topics_emb, (self.batch_size,self.sentence_length,self.topics_num))
        topics_emb = nd.softmax(topics_emb,axis=2)
        topics_emb = self.mlp_topic(topics_emb)
        word_emb = self.mlp_word(word_emb)
        xw = nd.concat(user_word, word_emb, topics_emb, dim=1)
        xw_1 = self.mlp_w1(xw)
        xw_2 = self.mlp_w2(xw_1)
        res = self.mlp(xw_2)
        return res

class ncf_topic_nt(nn.Block):
    def __init__(self, nb_users, topics_num, sentence_length, batch_size=1178, word_dim=128, visual_dim=2048, visual_emb=512, vocabulary_size=233243, **kwargs):
        super(ncf_topic_nt, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_uw = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.mlp_topic = nn.Dense(units=word_dim, activation='relu')
            self.mlp_w1 = nn.Dense(units=128, activation='relu')
            self.mlp_w2 = nn.Dense(units=64, activation='relu')
            self.mlp = nn.Dense(units=1, activation='sigmoid')
            self.topics_num = topics_num
            self.word_dim = word_dim 
            self.batch_size = batch_size
            self.sentence_length = sentence_length
    def forward(self, user_id, text, topics):
        user_word = self.emb_uw(user_id)
        word_emb = self.emb_word(text)
        topics_emb = self.emb_word(topics)
        topics_emb = nd.transpose(topics_emb, axes=(1,0))
        topics_emb = nd.reshape(topics_emb, (self.word_dim,self.topics_num,1))
        topics_emb = nd.dot(word_emb, topics_emb)
        topics_emb = nd.reshape(topics_emb, (self.batch_size,self.sentence_length,self.topics_num))
        topics_emb = nd.softmax(topics_emb,axis=2)
        topics_emb = self.mlp_topic(topics_emb)
        xw = nd.concat(user_word, topics_emb, dim=1)
        xw_1 = self.mlp_w1(xw)
        xw_2 = self.mlp_w2(xw_1)
        res = self.mlp(xw_2)
        return res

class ncf_vtopic(nn.Block):
    def __init__(self, nb_users, topics_num, sentence_length, batch_size=1178, word_dim=128, visual_dim=2048, visual_emb=512, vocabulary_size=233243, **kwargs):
        super(ncf_vtopic, self).__init__(**kwargs)
        with self.name_scope():
            self.dense_v = nn.Dense(units=visual_emb, activation='relu')
            self.emb_uv = nn.Embedding(nb_users, visual_emb, weight_initializer=mx.init.Uniform(1))
            self.mlp_v1 = nn.Dense(units=256, activation='relu')
            self.mlp_v2 = nn.Dense(units=64, activation='relu')
            ####################################################
            self.emb_uw = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.mlp_topic = nn.Dense(units=word_dim, activation='relu')
            self.mlp_w1 = nn.Dense(units=128, activation='relu')
            self.mlp_w2 = nn.Dense(units=64, activation='relu')
            ####################################################
            self.mlp = nn.Dense(units=1, activation='sigmoid')
            ####################################################
            self.topics_num = topics_num
            self.word_dim = word_dim 
            self.batch_size = batch_size
            self.sentence_length = sentence_length
    def forward(self, user_id, text, topics, visual):
        user_vis = self.emb_uv(user_id)
        visual_emb = self.dense_v(visual)
        xv = nd.concat(user_vis, visual_emb, dim=1)
        xv_1 = self.mlp_v1(xv)
        xv_2 = self.mlp_v2(xv_1)
        ################################
        user_word = self.emb_uw(user_id)
        word_emb = self.emb_word(text)
        topics_emb = self.emb_word(topics)
        topics_emb = nd.transpose(topics_emb, axes=(1,0))
        topics_emb = nd.reshape(topics_emb, (self.word_dim,self.topics_num,1))
        topics_emb = nd.dot(word_emb, topics_emb)
        topics_emb = nd.reshape(topics_emb, (self.batch_size,self.sentence_length,self.topics_num))
        topics_emb = nd.softmax(topics_emb,axis=2)
        topics_emb = self.mlp_topic(topics_emb)
        xt = nd.concat(user_word, topics_emb, dim=1)
        xt_1 = self.mlp_w1(xt)
        xt_2 = self.mlp_w2(xt_1)
        ################################
        x = nd.concat(xv_2, xt_2,dim=1)
        res = self.mlp(x)
        return res

class sncf(nn.Block):
    def __init__(self, nb_users, word_dim=128, visual_dim=2048, vocabulary_size=233243, **kwargs):
        super(sncf, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_uw = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')
            self.mlp_w1 = nn.Dense(units=128, activation='relu')
            self.mlp_w2 = nn.Dense(units=64, activation='relu')
            self.mlp = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, text):
        user_word = self.emb_uw(user_id)
        word_emb = self.emb_word(text)
        word_emb = self.mlp_word(word_emb)
        xw = nd.concat(user_word, word_emb, dim=1)
        xw_1 = self.mlp_w1(xw)
        xw_2 = self.mlp_w2(xw_1)
        res = self.mlp(xw_2)
        return res

class ncf_i(nn.Block):
    def __init__(self, nb_users, nb_photos, vocabulary_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_i, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_user  = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            # self.emb_photo = nn.Embedding(nb_photos, photo_dim, weight_initializer=mx.init.Uniform(1))
            # self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            # self.mlp_photo = nn.Dense(units=photo_dim, activation='relu')
            self.mlp_user = nn.Dense(units=word_dim, activation='relu')
            # self.mlp_word = nn.Dense(units=word_dim, activation='relu')
            self.mlp_1 = nn.Dense(units=128, activation='relu')
            self.mlp_2 = nn.Dense(units=64, activation='relu')
            self.mlp_3 = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, photo_id, user_doc, text):
        user_emb = self.emb_user(user_id)
        # photo_emb = self.emb_photo(photo_id)
        # text_emb = self.emb_word(text)
        # text_emb = self.mlp_word(text_emb)
        # photo_doc_emb = self.emb_photo(photo_doc)
        # photo_doc_emb = self.mlp_photo(photo_doc_emb)
        # x = nd.concat(user_emb, photo_doc_emb, photo_emb, text_emb, dim=1)
        user_doc_emb = self.emb_user(user_doc)
        user_doc_emb = self.mlp_user(user_doc_emb)
        # x = nd.concat(user_emb, user_doc_emb, text_emb, dim=1)
        x = nd.concat(user_emb, user_doc_emb, dim=1)
        x_1 = self.mlp_1(x)
        x_1 = self.dropout(x_1)
        x_2 = self.mlp_2(x_1)
        res = self.mlp_3(x_2)
        return res

class ncf_itt(nn.Block):
    def __init__(self, nb_users, nb_photos, vocabulary_size, word_dim=128, photo_dim=64, visual_dim=2048, **kwargs):
        super(ncf_itt, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_user  = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.emb_word  = nn.Embedding(vocabulary_size, word_dim, weight_initializer=mx.init.Uniform(1))
            self.dropout = nn.Dropout(rate=0.25)
            self.mlp_user = nn.Dense(units=word_dim, activation='relu')
            self.mlp_word = nn.Dense(units=word_dim, activation='relu')
            self.mlp_1 = nn.Dense(units=128, activation='relu')
            self.mlp_2 = nn.Dense(units=32, activation='sigmoid')
    def forward(self, user_id, user_doc_list, text_list):
        user_emb = self.emb_user(user_id)
        text_list_emb = self.emb_word(text_list)
        text_list_emb = self.mlp_word(text_list_emb)
        user_doc_list_emb = self.emb_user(user_doc_list)
        user_doc_list_emb = self.mlp_user(user_doc_list_emb)
        x = nd.concat(user_emb, user_doc_list_emb, text_list_emb, dim=1)
        x_1 = self.mlp_1(x)
        x_1 = self.dropout(x_1)
        res = self.mlp_2(x_1)
        return res

class ncf_rnn(nn.Block):
    def __init__(self, nb_users, word_dim=128, visual_dim=2048, vocabulary_size=233243, **kwargs):
        super(ncf_rnn, self).__init__(**kwargs)
        with self.name_scope():
            self.emb_u = nn.Embedding(nb_users, word_dim, weight_initializer=mx.init.Uniform(1))
            self.mlp_w1 = nn.Dense(units=128, activation='relu')
            self.mlp_w2 = nn.Dense(units=64, activation='relu')
            self.mlp = nn.Dense(units=1, activation='sigmoid')
    def forward(self, user_id, text):
        user_word = self.emb_u(user_id)
        xw = nd.concat(user_word, text, dim=1)
        xw_1 = self.mlp_w1(xw)
        xw_2 = self.mlp_w2(xw_1)
        res = self.mlp(xw_2)
        return res
