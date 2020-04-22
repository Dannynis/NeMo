import nemo.collections
import json
# NeMo's "core" package
import nemo
# NeMo's ASR collection
import nemo.collections.asr as nemo_asr

import os, librosa,tqdm
# Function to build a manifest
def build_manifest(transcripts_path, manifest_path, wav_path):
    data_dir = os.path.dirname(wav_path)
    print ('lookoing for wavs in:{}'.format(data_dir))
    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            for line in tqdm.tqdm_notebook(fin,total=len(os.listdir(data_dir))):
                # Lines look like this:
                # <s> transcript </s> (fileID)
                transcript = line.split(',')[1]
                transcript = transcript.lower().strip()

                file_name = line.split(',')[0].split('/')[1]
                audio_path = os.path.join(
                    data_dir, file_name)

                duration = librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }
                json.dump(metadata, fout)
                fout.write('\n')

import json

train_manifest = '/data/verbit_train_manifest.json'

test_manifest = '/data/verbit_test_manifest_small.json'
# build_manifest(test_transcripts, test_manifest, '/data/test/')
print("Test manifest created.")
print("******")

data_dir = '/data/verbit_azure_exp_q15x5/'

# Create our NeuralModuleFactory, which will oversee the neural modules.
neural_factory = nemo.core.NeuralModuleFactory(
    log_dir=data_dir + '/logdir',
    create_tb_writer=True)

logger = nemo.logging

import nemo
# --- Config Information ---#
from ruamel.yaml import YAML

config_path = '/tmp/pycharm_project_573/examples/asr/configs/quarts15x5_verbit_pers.yaml'
# config_path = '/data/verbit_azure_exp/quarts_verbit_pers.yaml'

yaml = YAML(typ='safe')

with open(config_path, 'rt', encoding='utf8') as file:
    params = yaml.load(file)

labels = params['labels']  # Vocab

print ('labels are: {}'.format(labels))

# --- Instantiate Neural Modules --- #

# Create training and test data layers (which load data) and data preprocessor
data_layer_train = nemo_asr.AudioToTextDataLayer.import_from_config(
    config_path,
    "AudioToTextDataLayer_train",
    overwrite_params={"manifest_filepath": train_manifest}
)  # Training datalayer

data_layer_test = nemo_asr.AudioToTextDataLayer.import_from_config(
    config_path,
    "AudioToTextDataLayer_eval",
    overwrite_params={"manifest_filepath": test_manifest}
)  # Eval datalayer

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor.import_from_config(
    config_path, "AudioToMelSpectrogramPreprocessor"
)

# Create the Jasper_4x1 encoder as specified, and a CTC decoder
encoder = nemo_asr.JasperEncoder.import_from_config(
    config_path, "JasperEncoder"
)

decoder = nemo_asr.JasperDecoderForCTC.import_from_config(
    config_path, "JasperDecoderForCTC",
    overwrite_params={"num_classes": len(labels)}
)

ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

# --- Assemble Training DAG --- #
audio_signal, audio_signal_len, transcript, transcript_len = data_layer_train()

processed_signal, processed_signal_len = data_preprocessor(
    input_signal=audio_signal,
    length=audio_signal_len)

encoded, encoded_len = encoder(
    audio_signal=processed_signal,
    length=processed_signal_len)

log_probs = decoder(encoder_output=encoded)
preds = greedy_decoder(log_probs=log_probs)  # Training predictions
loss = ctc_loss(
    log_probs=log_probs,
    targets=transcript,
    input_length=encoded_len,
    target_length=transcript_len)

# --- Assemble Validation DAG --- #
(audio_signal_test, audio_len_test,
 transcript_test, transcript_len_test) = data_layer_test()

processed_signal_test, processed_len_test = data_preprocessor(
    input_signal=audio_signal_test,
    length=audio_len_test)

encoded_test, encoded_len_test = encoder(
    audio_signal=processed_signal_test,
    length=processed_len_test)

log_probs_test = decoder(encoder_output=encoded_test)
preds_test = greedy_decoder(log_probs=log_probs_test)  # Test predictions
loss_test = ctc_loss(
    log_probs=log_probs_test,
    targets=transcript_test,
    input_length=encoded_len_test,
    target_length=transcript_len_test)


 #--- Create Callbacks --- #

# We use these imports to pass to callbacks more complex functions to perform.
from nemo.collections.asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch
from functools import partial

train_callback = nemo.core.SimpleLossLoggerCallback(
    # Notice that we pass in loss, predictions, and the transcript info.
    # Of course we would like to see our training loss, but we need the
    # other arguments to calculate the WER.
    tensors=[loss, preds, transcript, transcript_len],
    # The print_func defines what gets printed.
    print_func=partial(
        partial(monitor_asr_train_progress,tb_logger=neural_factory.tb_writer ),
        labels=labels),step_freq=100,
    tb_writer=neural_factory.tb_writer
    )

# We can create as many evaluation DAGs and callbacks as we want,
# which is useful in the case of having more than one evaluation dataset.
# In this case, we only have one.
eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_test, preds_test, transcript_test, transcript_len_test],
    user_iter_callback=partial(
        process_evaluation_batch, labels=labels),
    user_epochs_done_callback=process_evaluation_epoch,
    eval_step=500,  # How often we evaluate the model on the test set
    tb_writer=neural_factory.tb_writer
    )

checkpoint_saver_callback = nemo.core.CheckpointCallback(
    folder=data_dir+'/checkpoints',
    step_freq=500  # How often checkpoints are saved
    )

if not os.path.exists(data_dir+'/checkpoints'):
    os.makedirs(data_dir+'/checkpoints')


# --- Start Training! --- #
neural_factory.train(
    tensors_to_optimize=[loss],
    callbacks=[train_callback, eval_callback, checkpoint_saver_callback],
    optimizer='novograd',batches_per_step=1,
    optimization_params={
        "num_epochs": 100000, "lr": 0.01, "weight_decay": 1e-4
    })

# Training for 100 epochs will take a little while, depending on your machine.
# It should take about 20 minutes on Google Colab.
# At the end of 100 epochs, your evaluation WER should be around 20-25%.