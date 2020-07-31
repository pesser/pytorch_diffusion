import streamlit as st
import time
from pytorch_diffusion.diffusion import Diffusion


class tqdm(object):
    """
    tqdm-like progress bar for streamlit, adapted from
    https://github.com/streamlit/streamlit/issues/160#issuecomment-534385137
    """
    def __init__(self, iterable, total=None, pbar=None):
        if pbar is None:
            pbar = st.empty()
        self.prog_bar = pbar
        self.prog_bar.progress(0)
        self.iterable = iterable
        self.length = total if total is not None else len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)

@st.cache(allow_output_mutation=True)
def get_state(name, ema):
    if ema:
        name = "ema_"+name
    diffusion = Diffusion.from_pretrained(name)
    state = {"x": diffusion.denoise(1, n_steps=0),
             "curr_step": diffusion.num_timesteps,
             "diffusion": diffusion}
    return state

def main():
    st.title("Diffusion Model Demo")

    name = st.sidebar.radio("Model", ("cifar10", "lsun_bedroom", "lsun_cat", "lsun_church"))
    ema = st.sidebar.checkbox("ema", value=True)
    state = get_state(name, ema=ema)

    diffusion = state["diffusion"]
    st.text("Running {} model on {}".format(name, diffusion.device))

    clip = st.sidebar.checkbox("clip outputs", value=True)
    show_x0 = st.sidebar.checkbox("show predicted x0 during denoising", value=False)

    n_steps = st.sidebar.number_input("Number of steps",
                                      min_value=1,
                                      max_value=diffusion.num_timesteps,
                                      value=diffusion.num_timesteps)

    pbar = st.sidebar.empty()
    pbar.progress(0)
    def tqdm_factory(*args, **kwargs):
        return tqdm(*args, **kwargs, pbar=pbar)

    output = st.empty()
    step = st.empty()

    def callback(x, i, x0=None):
        if show_x0 and x0 is not None:
            x = x0
        output.image(diffusion.torch2hwcuint8(x, clip=clip)[0])
        step.text("Current step: {}".format(i))
    callback(state["x"], state["curr_step"])

    denoise = st.sidebar.button("Denoise")
    if state["curr_step"] > 0 and denoise:
        x = diffusion.denoise(1,
                              n_steps=n_steps, x=state["x"],
                              curr_step=state["curr_step"],
                              progress_bar=tqdm_factory,
                              callback=callback)
        state["x"] = x
        state["curr_step"] = max(0, state["curr_step"]-n_steps)

    diffuse = st.sidebar.button("Diffuse")
    if state["curr_step"] < diffusion.num_timesteps and diffuse:
        x = diffusion.diffuse(1,
                              n_steps=n_steps, x=state["x"],
                              curr_step=state["curr_step"],
                              progress_bar=tqdm_factory,
                              callback=callback)
        state["x"] = x
        state["curr_step"] = min(diffusion.num_timesteps, state["curr_step"]+n_steps)




if __name__ == "__main__":
    main()
