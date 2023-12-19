# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
import argparse
import sys
import os
import tensorflow as tf
import numpy as np
import re
from Classifier import *
from torchmetrics import CharErrorRate

FLAGS = None
parser = argparse.ArgumentParser()

parser.add_argument(
    '--graph', type=str, default=r'C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/speech_commands_v0.01/conv_actions_frozen.pb', help='Model to use for identification.')
parser.add_argument(
    '--labels', type=str, default=r'C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/speech_commands_v0.01/conv_actions_labels.txt', help='Path to file containing labels.')
parser.add_argument(
    '--input_name',
    type=str,
    default='wav_data:0',
    help='Name of WAVE data input node in model.')
parser.add_argument(
    '--output_name',
    type=str,
    default='labels_softmax:0',
    help='Name of node outputting a prediction in the model.')
parser.add_argument(
    '--how_many_labels',
    type=int,
    default=1,
    help='Number of results to show.')


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.io.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.compat.v1.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      # print('%s (score = %.5f)' % (human_string, score))
      
    return 0, human_string


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.io.gfile.exists(wav):
    raise ValueError('Audio file does not exist at {0}'.format(wav))
  if not labels or not tf.io.gfile.exists(labels):
    raise ValueError('Labels file does not exist at {0}'.format(labels))

  if not graph or not tf.io.gfile.exists(graph):
    raise ValueError('Graph file does not exist at {0}'.format(graph))

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  return run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)


# def exp():
#   """Entry point for script, converts flags to arguments."""

#   return predicted

def main():
    # folder = r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Diffusion_Output'
    # folder = r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Diffusion_Output'
    # folder = r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\output_10_500_20\result\down\go'
    # fdir = os.listdir(folder) # All files in current directory
    
    # All files
    # fs = [f for f in fdir if f.endswith('wav')] # Find soundfiles
    # Sort
    def extract_number(s):
        return int(re.search(r'\d+', s).group())

    # Use the custom sorting key
    # fs = sorted(fs, key=extract_number)

    index = 0
    table1 = 0
    table2 = 0
    
    
    # Sort
    def extract_number(s):
        return int(re.search(r'\d+', s).group())

    word_index = -1
    # Classifier:
    model_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.pbmm'
    scorer_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.scorer'
    classifier = DeepSpeechTranscriber(model_path, scorer_path)
    
    cer = CharErrorRate()

    original = []
    attacked = []
    


    folder = r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Diffusion_Output'
    fdir = os.listdir(folder) # All files in current directory
        
    # All files
    fs = [f for f in fdir if f.endswith('wav')] # Find soundfiles
        
    # Use the custom sorting key
    fs = sorted(fs, key=extract_number)
    to_pop = "down"
    words = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
    words.remove(to_pop)
    n = 0
    word_index = -1
    for i in fs:
        
        if n % 20 == 0:
            word_index += 1
        current_word = words[word_index]

        audio_file = folder + "\\" + f'{n}.wav'
        
        # BB:
        # FLAGS, unparsed = parser.parse_known_args()

        # zero, transcription = label_wav(audio_file, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            # FLAGS.output_name, FLAGS.how_many_labels)
            
        # WB:
        transcription = classifier.transcribe_audio_file(audio_file)
        
        if transcription == current_word:
            print(f'{transcription}: Correct')
            table1 += 1
        elif transcription == to_pop:
            print(f'{transcription}: Wrong')    
            table2 += 1
        else:
            print(f'{transcription}: Wrong') 
        n += 1
        print(n)
        print(word_index)
        # original.append(current_word)
        # attacked.append(transcription)
    # print(table1 / len(fs))
    # print(table2)
    return table1 / len(fs), table2 / len(fs)
        # return table1 / len(fs)
        # return 1 - cer(original, attacked)
print(main())
# FLAGS, unparsed = parser.parse_known_args()
# zero, predicted = tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
# print(main())
# if __name__ == "__main__":
#     zero, transcription = label_wav(r'C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/Diffusion_Output/102.wav', FLAGS.labels, FLAGS.graph, FLAGS.input_name,
#           FLAGS.output_name, FLAGS.how_many_labels)