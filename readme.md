
# Fork notes

This is a fork of a fork of a handwriting synthesiser combined with a [postcard text generator](https://gitlab.com/scouarn/postcard_generator).

Requirements:

- Python between 3.7 and 3.9 (I recommend using a [venv](https://docs.python.org/3/library/venv.html))
- A HuggingFace token in the `HF_TOKEN` env variable.
- `cerevoice/txt2wav` (from `cerevoice_sdk_.../examples/basictts/txt2wav`)
- `cerevoice/voice.voice` and `cerevoice/license.lic`


Usage:

```shell
$ python postcard_generator.py
```


![](img/banner.svg)
# Handwriting Synthesis
Implementation of the handwriting synthesis experiments in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves.  The implementation closely follows the original paper, with a few slight deviations, and the generated samples are of similar quality to those presented in the paper.

Web demo from original github (which I forked) is available <a href="https://seanvasquez.com/handwriting-generation/">here</a>.

## Fork Notes
Fork has the following changes from the original
1. Updated to work with TensorFlow 2.15.0 (current as of December 2023), but uses the v1 compat mechanism.  Honestly I hacked and burned through many errors in an hour and a half (I forked at 7:05 PM and am writing this at 8:35 PM) and just verified by running the demo.py and looking through the output image files.  It didn't create a banner.svg file (no idea if it was supposed to), and it gets a lot of deprecation warnings so use at your own risk.  TensorFlow will likely abandon some of this V1 compat stuff in the relatively near future, but this should work as long as you can still get 2.15.0 for whatever python version you have.
2. I'm going to split the Hand class into its own file (per the original author's suggestion).
3. I left everything else below this text alone (except striking the "split Hand class" request).
4. If you think my fork is super sloppy (you're right) and want to do it right- the major job is to convert the deprecated `tf.nn.rnn_cell.LSTMCell` and replace it with `tf.keras.layers.LSTMCell`.  It is not a drop in replacement.  Have at it.

## Usage
```python
lines = [
    "Now this is a story all about how",
    "My life got flipped turned upside down",
    "And I'd like to take a minute, just sit right there",
    "I'll tell you how I became the prince of a town called Bel-Air",
]
biases = [.75 for i in lines]
styles = [9 for i in lines]
stroke_colors = ['red', 'green', 'black', 'blue']
stroke_widths = [1, 2, 1, 2]

hand = Hand()
hand.write(
    filename='img/usage_demo.svg',
    lines=lines,
    biases=biases,
    styles=styles,
    stroke_colors=stroke_colors,
    stroke_widths=stroke_widths
)
```
![](img/usage_demo.svg)

~~Currently, the `Hand` class must be imported from `demo.py`.  If someone would like to package this project to make it more usable, please [contribute](#contribute).~~

A pretrained model is included, but if you'd like to train your own, read <a href='https://github.com/sjvasquez/handwriting-synthesis/tree/master/data/raw'>these instructions</a>.

## Demonstrations
Below are a few hundred samples from the model, including some samples demonstrating the effect of priming and biasing the model.  Loosely speaking, biasing controls the neatness of the samples and priming controls the style of the samples. The code for these demonstrations can be found in `demo.py`.

### Demo #1:
The following samples were generated with a fixed style and fixed bias.

**Smash Mouth – All Star (<a href="https://www.azlyrics.com/lyrics/smashmouth/allstar.html">lyrics</a>)**
![](img/all_star.svg)

### Demo #2
The following samples were generated with varying style and fixed bias.  Each verse is generated in a different style.

**Vanessa Carlton – A Thousand Miles (<a href="https://www.azlyrics.com/lyrics/vanessacarlton/athousandmiles.html">lyrics</a>)**
![](img/downtown.svg)

### Demo #3
The following samples were generated with a fixed style and varying bias.  Each verse has a lower bias than the previous, with the last verse being unbiased.

**Leonard Cohen – Hallelujah (<a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">lyrics</a>)**
![](img/give_up.svg)

## Contribute
This project was intended to serve as a reference implementation for a research paper, but since the results are of decent quality, it may be worthwile to make the project more broadly usable.  I plan to continue focusing on the machine learning side of things.  That said, I'd welcome contributors who can:

  - Package this, and otherwise make it look more like a usable software project and less like research code.
  - Add support for more sophisticated drawing, animations, or anything else in this direction.  Currently, the project only creates some simple svg files.
