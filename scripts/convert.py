import torch
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pprint import pprint
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from pytorch_diffusion.model import (
    get_timestep_embedding,
    nonlinearity,
    Normalize,
    Upsample,
    Downsample,
    ResnetBlock,
    AttnBlock,
    Model,
)

try:
    from diffusion_tf.models.unet import (
        nonlinearity as nonlinearity_tf_,
        normalize as normalize_tf_,
        upsample as upsample_tf_,
        downsample as downsample_tf_,
        resnet_block as resnet_block_tf_,
        attn_block as attn_block_tf_,
        model as model_tf_,
    )
    from diffusion_tf.nn import (
        get_timestep_embedding as get_timestep_embedding_tf_,
    )
except ImportError as e:
    raise ImportError("Clone https://github.com/hojonathanho/diffusion and add it to PYTHONPATH.") from e




def get_tf_var(scope, name):
    if scope.startswith("ExponentialMovingAverage:0/"):
        scope = scope[len("/ExponentialMovingAverage:0"):]
        name = name+"/ExponentialMovingAverage"
    return tf.get_default_session().run(tf.get_default_graph().get_tensor_by_name(scope+"/"+name+":0"))


def copy_normalize(torchlayer, tfscope):
    with torch.no_grad():
        torchlayer.bias.copy_(torch.tensor(get_tf_var(tfscope, "beta")))
        torchlayer.weight.copy_(torch.tensor(get_tf_var(tfscope, "gamma")))


def copy_conv(torchlayer, tfscope):
    with torch.no_grad():
        torchlayer.bias.copy_(torch.tensor(get_tf_var(tfscope, "b")))
        torchlayer.weight.copy_(torch.tensor(get_tf_var(tfscope, "W")).permute(3,2,0,1))


def copy_nin(torchlayer, tfscope):
    with torch.no_grad():
        torchlayer.bias.copy_(torch.tensor(get_tf_var(tfscope, "b")))
        torchlayer.weight.copy_(torch.tensor(get_tf_var(tfscope, "W")).permute(1,0)[:,:,None,None])


def copy_dense(torchlayer, tfscope):
    with torch.no_grad():
        torchlayer.bias.copy_(torch.tensor(get_tf_var(tfscope, "b")))
        torchlayer.weight.copy_(torch.tensor(get_tf_var(tfscope, "W")).permute(1,0))


def copy_upsample(torchlayer, tfscope):
    copy_conv(torchlayer.conv, tfscope+"/conv")


def copy_downsample(torchlayer, tfscope):
    copy_conv(torchlayer.conv, tfscope+"/conv")


def copy_resnet_block(torchlayer, tfscope):
    copy_conv(torchlayer.conv1, tfscope+"/conv1")
    copy_conv(torchlayer.conv2, tfscope+"/conv2")
    copy_normalize(torchlayer.norm1, tfscope+"/norm1")
    copy_normalize(torchlayer.norm2, tfscope+"/norm2")
    copy_dense(torchlayer.temb_proj, tfscope+"/temb_proj")
    if hasattr(torchlayer, "conv_shortcut"):
        copy_conv(torchlayer.conv_shortcut, tfscope+"/conv_shortcut")
    elif hasattr(torchlayer, "nin_shortcut"):
        copy_nin(torchlayer.nin_shortcut, tfscope+"/nin_shortcut")


def copy_attn_block(torchlayer, tfscope):
    copy_normalize(torchlayer.norm, tfscope+"/norm")
    copy_nin(torchlayer.q, tfscope+"/q")
    copy_nin(torchlayer.k, tfscope+"/k")
    copy_nin(torchlayer.v, tfscope+"/v")
    copy_nin(torchlayer.proj_out, tfscope+"/proj_out")


def copy_model(model, tfscope):
    copy_dense(model.temb.dense[0], tfscope+"/temb/dense0")
    copy_dense(model.temb.dense[1], tfscope+"/temb/dense1")
    copy_conv(model.conv_in, tfscope+"/conv_in")
    for i_level in range(model.num_resolutions):
        for i_block in range(model.num_res_blocks):
            copy_resnet_block(model.down[i_level].block[i_block],
                              tfscope+"/down_{}/block_{}".format(i_level,
                                                                i_block))
            if len(model.down[i_level].attn) > 0:
                copy_attn_block(model.down[i_level].attn[i_block],
                                tfscope+"/down_{}/attn_{}".format(i_level,
                                                                 i_block))
        if i_level != model.num_resolutions-1:
            copy_downsample(model.down[i_level].downsample,
                            tfscope+"/down_{}/downsample".format(i_level))

    copy_resnet_block(model.mid.block_1, tfscope+"/mid/block_1")
    copy_attn_block(model.mid.attn_1, tfscope+"/mid/attn_1")
    copy_resnet_block(model.mid.block_2, tfscope+"/mid/block_2")

    for i_level in reversed(range(model.num_resolutions)):
        for i_block in range(model.num_res_blocks+1):
            copy_resnet_block(model.up[i_level].block[i_block],
                              tfscope+"/up_{}/block_{}".format(i_level,
                                                               i_block))
            if len(model.up[i_level].attn) > 0:
                copy_attn_block(model.up[i_level].attn[i_block],
                                tfscope+"/up_{}/attn_{}".format(i_level,
                                                                i_block))
        if i_level != 0:
            copy_upsample(model.up[i_level].upsample,
                          tfscope+"/up_{}/upsample".format(i_level))

    copy_normalize(model.norm_out, tfscope+"/norm_out")
    copy_conv(model.conv_out, tfscope+"/conv_out")


def show_torchparams(torchlayer):
    vs = dict(torchlayer.named_parameters())
    pprint(dict((k, tuple(int(s) for s in v.shape)) for k,v in vs.items()))


def show_tfparams(scope):
    vs = dict((v.name, v) for v in tf.global_variables() if v.name.startswith(scope))
    pprint(dict((k, tuple(int(s) for s in v.shape)) for k,v in vs.items()))


def reinit_tf(scope, scale=1.0):
    # reinit everything normal to avoid missing wrong computations due to zero init
    vs = dict((v.name, v) for v in tf.global_variables() if v.name.startswith(scope))
    for k,v in vs.items():
        tf.get_default_session().run(v.assign(scale*tf.random_normal(shape=v.shape,
                                                                     dtype=v.dtype.base_dtype)))


def check(ftorch, ftf, x, is_image=True, **kwargs):
    xtorch = torch.tensor(x)
    kwargstorch = dict((k, torch.tensor(v)) for k,v in kwargs.items())
    if is_image:
        xtorch = xtorch.permute(0,3,1,2)
    outtorch = ftorch(xtorch, **kwargstorch)
    if is_image:
        outtorch = outtorch.permute(0,2,3,1)
    outtorch = outtorch.detach().numpy()

    outtf = ftf(x, **kwargs)

    allclose = np.allclose(outtorch, outtf, rtol=1e-3, atol=1e-2)
    if not allclose:
        diff = outtorch-outtf
        reldiff = np.linalg.norm(diff) / np.linalg.norm(outtf)
        raise Exception("Relative Error: {}".format(reldiff))


def test_ops():
    """Compare ops under random init and random inputs."""
    # test inputs
    x = (np.random.randn(2,8,8,64).astype(np.float32)+1.0)**2
    b,h,w,c = x.shape
    temb = (np.random.randn(b,512).astype(np.float32)-1.0)**3
    ts = np.random.randint(10000, size=(b,))

    with tf.Session() as s:
        xph = tf.placeholder(dtype=tf.float32, shape=x.shape)
        tembph = tf.placeholder(dtype=tf.float32, shape=temb.shape)
        tsph = tf.placeholder(dtype=tf.int32, shape=ts.shape)


        print("nonlinearity")
        nonlinearity_tf_op = nonlinearity_tf_(xph)
        nonlinearity_tf = lambda x: s.run(nonlinearity_tf_op, feed_dict={xph: x})
        check(nonlinearity, nonlinearity_tf, x)

        print("get_timestep_embedding")
        get_timestep_embedding_tf_op = get_timestep_embedding_tf_(tsph, 512)
        get_timestep_embedding_tf = lambda t: s.run(get_timestep_embedding_tf_op, feed_dict={tsph: t})
        check(lambda t: get_timestep_embedding(t, 512), get_timestep_embedding_tf, ts, is_image=False)

        print("normalize")
        normalize = Normalize(c)
        show_torchparams(normalize)

        normalize_tf_op = normalize_tf_(xph, temb=None, name="normalize")
        normalize_tf = lambda x: s.run(normalize_tf_op, feed_dict={xph: x})
        reinit_tf("normalize")
        show_tfparams("normalize")

        copy_normalize(normalize, "normalize")
        check(normalize, normalize_tf, x)

        print("upsample")
        upsample = Upsample(c, True)
        show_torchparams(upsample)

        upsample_tf_op = upsample_tf_(xph, name="upsample", with_conv=True)
        upsample_tf = lambda x: s.run(upsample_tf_op, feed_dict={xph: x})
        reinit_tf("upsample")
        show_tfparams("upsample")

        copy_upsample(upsample, "upsample")
        check(upsample, upsample_tf, x)


        print("downsample")
        downsample = Downsample(c, True)
        show_torchparams(downsample)

        downsample_tf_op = downsample_tf_(xph, name="downsample", with_conv=True)
        downsample_tf = lambda x: s.run(downsample_tf_op, feed_dict={xph: x})
        reinit_tf("downsample")
        show_tfparams("downsample")

        copy_downsample(downsample, "downsample")
        check(downsample, downsample_tf, x)


        print("resnet_block")
        resnet_block = ResnetBlock(in_channels=c, dropout=0)
        show_torchparams(resnet_block)

        resnet_block_tf_op = resnet_block_tf_(xph, name="resnet_block", temb=tembph,
                                              dropout=0)
        resnet_block_tf = lambda x, temb: s.run(resnet_block_tf_op,
                                                feed_dict={xph: x, tembph: temb})
        reinit_tf("resnet_block")
        show_tfparams("resnet_block")

        copy_resnet_block(resnet_block, "resnet_block")
        check(resnet_block, resnet_block_tf, x, temb=temb)


        print("exp_resnet_block")
        out_channels = 128
        resnet_block = ResnetBlock(in_channels=c, out_channels=out_channels, dropout=0)
        show_torchparams(resnet_block)

        resnet_block_tf_op = resnet_block_tf_(xph, out_ch=out_channels,
                                              name="exp_resnet_block", temb=tembph,
                                              dropout=0)
        resnet_block_tf = lambda x, temb: s.run(resnet_block_tf_op,
                                                feed_dict={xph: x, tembph: temb})
        reinit_tf("exp_resnet_block")
        show_tfparams("exp_resnet_block")

        copy_resnet_block(resnet_block, "exp_resnet_block")
        check(resnet_block, resnet_block_tf, x, temb=temb)


        print("attn_block")
        attn_block = AttnBlock(in_channels=c)
        show_torchparams(attn_block)

        attn_block_tf_op = attn_block_tf_(xph, name="attn_block", temb=None)
        attn_block_tf = lambda x: s.run(attn_block_tf_op, feed_dict={xph: x})
        reinit_tf("attn_block")
        show_tfparams("attn_block")

        copy_attn_block(attn_block, "attn_block")
        check(attn_block, attn_block_tf, x)


        print("model")
        x = 2*np.random.rand(1,32,32,3).astype(np.float32) - 1.0
        b,h,w,c = x.shape
        ts = np.random.randint(1000, size=(b,)).astype(np.int32)
        xph = tf.placeholder(dtype=tf.float32, shape=x.shape)
        tsph = tf.placeholder(dtype=tf.int32, shape=ts.shape)

        model = Model(resolution=32, in_channels=3, out_ch=3, ch=128, ch_mult=(1,2,2,2),
                      num_res_blocks=2, attn_resolutions=(16,), dropout=0.0)
        show_torchparams(model)

        model_tf_op = model_tf_(xph, t=tsph, y=None, name="model", num_classes=1,
                                ch=128, out_ch=3, ch_mult=(1,2,2,2),
                                num_res_blocks=2, attn_resolutions=(16,))
        model_tf = lambda x, t: s.run(model_tf_op, feed_dict={xph: x, tsph: t})
        reinit_tf("model", scale=1e-3) # downscaled to avoid nan
        show_tfparams("model")

        copy_model(model, "model")
        check(model, model_tf, x, t=ts)
    tf.reset_default_graph()


def sample_tf(bs=1, nb=1, which=None):
    import tqdm
    # add diffusion/scripts to PYTHONPATH, too
    from diffusion_tf.tpu_utils.tpu_utils import make_ema
    import diffusion_tf.utils as utils
    from scripts.run_cifar import Model as cifar10_model
    from scripts.run_lsun import Model as lsun_model
    from diffusion_tf.diffusion_utils_2 import get_beta_schedule
    import PIL.Image

    ckpts = {
        "cifar10": "diffusion_models_release/diffusion_cifar10_model/model.ckpt-790000",
        "lsun_bedroom": "diffusion_models_release/diffusion_lsun_bedroom_model/model.ckpt-2388000",
        "lsun_cat": "diffusion_models_release/diffusion_lsun_cat_model/model.ckpt-1761000",
        "lsun_church": "diffusion_models_release/diffusion_lsun_church_model/model.ckpt-4432000",
    }
    models = {
        "cifar10": cifar10_model,
        "lsun_bedroom": lsun_model,
        "lsun_cat": lsun_model,
        "lsun_church": lsun_model,
    }
    betas = get_beta_schedule(beta_schedule="linear", beta_start=0.0001,
                              beta_end=0.02, num_diffusion_timesteps=1000)
    cifar10_config = {
        "model_name": "unet2d16b2",
        "model_mean_type": "eps",
        "model_var_type": "fixedlarge",
        "betas": betas,
        "loss_type": "noisepred",
        "num_classes": 1,
        "dropout": 0,
        "randflip": 0,
    }
    lsun_config = {
        "model_name": "unet2d16b2c112244",
        "betas": betas,
        "loss_type": "noisepred",
        "num_classes": 1,
        "dropout": 0,
        "randflip": 0,
        "block_size": 1,
    }
    model_configs = {
        "cifar10": cifar10_config,
        "lsun_bedroom": lsun_config,
        "lsun_cat": lsun_config,
        "lsun_church": lsun_config,
    }
    img_shapes = {
        "cifar10": (32,32,3),
        "lsun_bedroom": (256,256,3),
        "lsun_cat": (256,256,3),
        "lsun_church": (256,256,3),
    }

    which = which if which is not None else ["cifar10", "lsun_bedroom", "lsun_cat", "lsun_church"]
    if type(which) == str:
        which = [which]

    for name in which:
        os.makedirs("results/tf_{}".format(name), exist_ok=True)
        print("Writing tf samples in {}".format("results/tf_{}".format(name)))
        ema = name.startswith("ema_")
        basename = name[len("ema_"):] if ema else name
        with tf.Session() as sess:
            print("Loading {} model".format(name))
            model = models[basename](**model_configs[basename])
            # build graph
            x_ = tf.fill([bs, *img_shapes[basename]], value=np.nan)
            y = tf.fill([bs,], value=0)

            sample = model.samples_fn(x_, y)

            global_step = tf.train.get_or_create_global_step()
            if ema:
                ema_, _ = make_ema(global_step=global_step,
                                   ema_decay=1e-10,
                                   trainable_variables=tf.trainable_variables())
                with utils.ema_scope(ema_):
                  print('===== EMA SAMPLES =====')
                  sample = model.progressive_samples_fn(x_, y)

            # load ckpt
            ckpt = ckpts[basename]
            print('restoring')
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)
            global_step_val = sess.run(global_step)
            print('restored global step: {}'.format(global_step_val))
            for ib in tqdm.tqdm(range(nb), desc="Batch"):
                # test sampling
                result = sess.run(sample)
                samples = result["samples"]
                for i in range(samples.shape[0]):
                    np_sample = ((samples[i]+1.0)*127.5).astype(np.uint8)
                    PIL.Image.fromarray(np_sample).save("results/tf_{}/{:06}.png".format(name,
                                                                                         ib*bs+i))
        tf.reset_default_graph()



def transplant(cfg, tf_ckpt, torch_ckpt, ema=False):
    with tf.Session() as s:
        x = np.random.rand(1,cfg["resolution"],cfg["resolution"],cfg["in_channels"]).astype(np.float32)
        x = 2.0*x-1.0
        b,h,w,c = x.shape
        ts = np.random.randint(1000, size=(b,)).astype(np.int32)
        xph = tf.placeholder(dtype=tf.float32, shape=x.shape)
        tsph = tf.placeholder(dtype=tf.int32, shape=ts.shape)

        model = Model(resolution=cfg["resolution"],
                      in_channels=cfg["in_channels"],
                      out_ch=cfg["out_ch"],
                      ch=cfg["ch"],
                      ch_mult=cfg["ch_mult"],
                      num_res_blocks=cfg["num_res_blocks"],
                      attn_resolutions=cfg["attn_resolutions"],
                      dropout=0.0)
        show_torchparams(model)

        model_tf_op = model_tf_(xph, t=tsph, y=None, name="model", num_classes=1,
                                ch=cfg["ch"],
                                out_ch=cfg["out_ch"],
                                ch_mult=cfg["ch_mult"],
                                num_res_blocks=cfg["num_res_blocks"],
                                attn_resolutions=cfg["attn_resolutions"],
                                dropout=0.0)
        model_tf = lambda x, t: s.run(model_tf_op, feed_dict={xph: x, tsph: t})

        global_step = tf.train.get_or_create_global_step()
        if ema:
            from diffusion_tf.tpu_utils.tpu_utils import make_ema
            import diffusion_tf.utils as utils
            ema_, _ = make_ema(global_step=global_step,
                               ema_decay=1e-10,
                               trainable_variables=tf.trainable_variables())
            scope = lambda: utils.ema_scope(ema_)
        else:
            import contextlib
            scope = lambda: contextlib.nullcontext()
        with scope():
            if ema:
                # redefine model op with ema variables
                model_tf_op = model_tf_(xph, t=tsph, y=None, name="model", num_classes=1,
                                        ch=cfg["ch"],
                                        out_ch=cfg["out_ch"],
                                        ch_mult=cfg["ch_mult"],
                                        num_res_blocks=cfg["num_res_blocks"],
                                        attn_resolutions=cfg["attn_resolutions"],
                                        dropout=0.0)
                model_tf = lambda x, t: s.run(model_tf_op, feed_dict={xph: x, tsph: t})
            show_tfparams("model")
            # restore tf ckpt
            saver = tf.train.Saver()
            saver.restore(s, tf_ckpt)
            global_step_val = s.run(global_step)
            print('restored global step: {}'.format(global_step_val))
            print("copying into pytorch model")
            tfscope = "model"
            if ema:
                tfscope = "ExponentialMovingAverage:0/"+tfscope
            copy_model(model, tfscope)
            print("checking")
            check(model, model_tf, x, t=ts)
            print("saving torch weights in {}".format(torch_ckpt))
            torch.save(model.state_dict(), torch_ckpt)
    tf.reset_default_graph()


def _transplant_model(name, step, cfg, ema=False,
                      tf_root="diffusion_models_release",
                      torch_root="diffusion_models_converted"):
    if not os.path.exists(tf_root):
        sys.exit("Download tf checkpoints from https://www.dropbox.com/sh/pm6tn31da21yrx4/AABWKZnBzIROmDjGxpB6vn6Ja and put them into `{}`".format(tf_root))
    tf_ckpt = os.path.join(tf_root,
                           "diffusion_{}_model/model.ckpt-{}".format(name,
                                                                     step))
    ema_prefix = "ema_" if ema else ""
    torch_dir = os.path.join(torch_root, ema_prefix+"diffusion_{}_model".format(name))
    os.makedirs(torch_dir, exist_ok=True)
    torch_ckpt = os.path.join(torch_dir, "model-{}.ckpt".format(step))
    transplant(cfg=cfg, tf_ckpt=tf_ckpt, torch_ckpt=torch_ckpt, ema=ema)


def transplant_cifar10(ema=False):
    cfg = {
        "resolution": 32,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1,2,2,2),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
    }
    _transplant_model(name="cifar10", step="790000", cfg=cfg, ema=ema)


def transplant_lsun_bedroom(ema=False):
    cfg = {
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1,1,2,2,4,4),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
    }
    _transplant_model(name="lsun_bedroom", step="2388000", cfg=cfg, ema=ema)


def transplant_lsun_cat(ema=False):
    cfg = {
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1,1,2,2,4,4),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
    }
    _transplant_model(name="lsun_cat", step="1761000", cfg=cfg, ema=ema)


def transplant_lsun_church(ema=False):
    cfg = {
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1,1,2,2,4,4),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
    }
    _transplant_model(name="lsun_church", step="4432000", cfg=cfg, ema=ema)
