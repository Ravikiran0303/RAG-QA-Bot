__author__ = 'Utsab_Chatterjee'

import ast

import datetime

import glob
import json

import os

import random

import re

import uuid

from pathlib import Path
from sentence_transformers import CrossEncoder
import azure.cognitiveservices.speech as speech_sdk

import numpy as np

import pandas as pd

from dotenv import load_dotenv

from google.cloud.dialogflowcx_v3beta1 import AgentsClient, SessionsClient, PlaybooksClient, FlowsClient, PagesClient

from google.cloud.dialogflowcx_v3beta1.types import session, audio_config

from google.cloud.dialogflowcx_v3beta1.types.flow import ListFlowsRequest

from google.cloud.dialogflowcx_v3beta1.types.page import ListPagesRequest

from google.cloud.dialogflowcx_v3beta1.types.playbook import ListPlaybooksRequest

from google.protobuf.json_format import MessageToDict

from ssml_builder.core import Speech

from BackgroundNoise import BackgroundNoise
# import noisereduce as nr

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "private_key.json"


class DialogflowCXAssurance:


    load_dotenv("./cx_env.env")

    PROJECT_ID = os.getenv("project_id")

    LOCATION_ID = os.getenv("location_id")

    AGENT_ID = os.getenv("agent_id")


    def __init__(self):
        self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
        self.agent = f"projects/{self.PROJECT_ID}/locations/{self.LOCATION_ID}/agents/{self.AGENT_ID}"

        # Language Code

        self.language_code = "en-us"


    def main(self):
        try:

            # reading test cases

            test_file_name = input("Enter test data file with extension : ")

            sheet_list = list(map(str, input("Enter sheet names in comma separated format: ").split(sep=',')))

            # Output configuration

            output_file_name = input("Enter output file name : ")

            parameter_set = input("Is parameter needs to be set ? (Y/N): ")

            api_invoke_required = input("Is Rest call needs to be invoked ? (Y/N): ")

            is_common_threshold = input("Do you want to set a common threshold value (Y/N): ")

            common_threshold_value = None

            if is_common_threshold.lower() == 'y':
                common_threshold_value = float(input("Enter common threshold value (0-100): "))

            date = datetime.datetime.now().strftime("%m_%d_%Y")

            file_name = "test_result_{}_{}.xlsx".format(output_file_name, date)

            writer = pd.ExcelWriter('./Test_Result/' + file_name, engine='openpyxl')

            summary_dict = {'Type': [], 'Total_Record': [], 'PASS': [], 'FAIL': [], 'PASS(%)': []}

            client_options = self.get_client_options()

            for sheet_name in sheet_list:

                # Temporary call

                # from API_Playbook_call import ApiCall

                # status_api = ApiCall().main()

                # if not status_api:

                #     print("backend data has not been set, calling api again...")

                #     status_api = ApiCall().main()

                print(f'==========Execution started for {sheet_name}==========')

                playbook_url = None

                page_url = None

                flow_or_playbook = int(input("1. Flow-Page\n2. Playbook\n3. Default\nChoose the test option: "))

                if flow_or_playbook == 1:

                    flow_dict = self.get_flows(client_options)

                    idx_map_flow = {key: i for i, key in enumerate(flow_dict)}

                    for key in flow_dict.keys():
                        print(str(idx_map_flow.get(key) + 1) + ". " + key)

                    choose_flow = int(input("Choose the flow: "))

                    flow_name = list(filter(lambda key: idx_map_flow[key] == choose_flow - 1, idx_map_flow))[0]

                    flow_id = flow_dict[flow_name]

                    page_dict = self.get_pages(client_options, flow_id)

                    idx_map_page = {key: i for i, key in enumerate(page_dict)}

                    for key in page_dict.keys():
                        print(str(idx_map_page.get(key) + 1) + ". " + key)

                    choose_page = int(input("Choose the page: "))

                    page_name = list(filter(lambda key: idx_map_page[key] == choose_page - 1, idx_map_page))[0]

                    page_url = page_dict[page_name]

                elif flow_or_playbook == 2:

                    playbook_dict = self.get_playbooks(client_options)

                    idx_map = {key: i for i, key in enumerate(playbook_dict)}

                    for key in playbook_dict.keys():
                        print(str(idx_map.get(key) + 1) + ". " + key)

                    choose_playbook = int(input("Choose the playbook: "))

                    playbook_name = list(filter(lambda key: idx_map[key] == choose_playbook - 1, idx_map))[0]

                    playbook_url = playbook_dict[playbook_name]


                else:

                    pass

                if common_threshold_value is None:
                    common_threshold_value = float(input("Enter common threshold value (0-100): "))

                data = pd.read_excel("./Test_Data/" + test_file_name, engine="openpyxl", sheet_name=sheet_name)
                data_dict = data.to_dict(orient='list')
                final_df = pd.DataFrame()
                run_audio = input(f"Do you want to run using audio for {sheet_name}?(Y/N) : ")
                numeric_match = input("If you want to test any numeric match (ID specific) ? (Y/N): ")

                if run_audio.lower() == 'y':
                    if 'Audio_file' in data_dict.keys():
                        absolute_path = input("Enter the parent audio path : ")
                        path = absolute_path + "\\" + sheet_name
                        response_with_audio = data_dict
                    else:
                        audio_type = int(input("1. Google Cloud\n2. Azure Cloud\nChoose the audio provider: "))
                        response_with_audio, absolute_path, path = None, None, None
                        if audio_type == 1:
                            response_with_audio, absolute_path, path = self.synthesize_text_gcp(data_dict, sheet_name)
                        elif audio_type == 2:
                            response_with_audio, absolute_path, path = self.synthesize_text_azure(data_dict, sheet_name)
                    main_audios = sorted(glob.glob(path + '/*.wav'), key=os.path.getctime)
                    bg_noise_addition = input("Do you want to add background noise (Y/N) : ")
                    if bg_noise_addition.lower() == 'y':
                        noises = glob.glob('./audio_noises/*')
                        noise_list = random.choices(noises, k=len(main_audios))
                        for i in range(len(main_audios)):
                            path_main = Path(main_audios[i])
                            path_audio = path_main.relative_to(os.path.abspath(".\\"))
                            obj = BackgroundNoise().add_background_noise(response_with_audio, path_audio,
                                                                     noise_list[i].split('\\')[1], i, path)
                            response_with_audio['Utterance'].extend(obj['Utterance'])
                            response_with_audio['Response'].extend(obj['Response'])
                            response_with_audio['Intent'].extend(obj['Intent'])
                            response_with_audio['Entity'].extend(obj['Entity'])
                            response_with_audio['Levels'].extend(obj['Levels'])
                            response_with_audio['Audio_file'].extend(obj['Audio_file'])
                            # print(response_with_audio)
                        print("Background noise addition completed")

                    # Calling intent detection using audio
                    test_data = response_with_audio['Utterance']
                    bot_response = response_with_audio['Response']
                    expected_intents = response_with_audio['Intent']
                    expected_entities = response_with_audio['Entity']
                    levels = response_with_audio['Levels']
                    audio_files = response_with_audio['Audio_file']
                    custom_audio_ip = input("Is Custom STT model required (y/n): ")
                    custom_model_type = None
                    if custom_audio_ip == 'y':
                        custom_model_type = input(
                            """1. Custom DOB Model\n2. Custom MID Model\n3. Custom Currency Model\n4. Custom SSN Model\n5. Custom Phone Model\n6. Custom NPI model\n7. Custom TIN model\nEnter your choice: """)

                    for i in range(len(test_data)):
                        session_id = uuid.uuid4()
                        data = {'test_data': test_data[i], 'bot_response': bot_response[i],
                                'expected_intent': expected_intents[i], 'expected_entity': expected_entities[i],
                                'levels': levels[i], 'audio_file': audio_files[i],
                                'expected_threshold': common_threshold_value}
                        result_df = self.detect_intent_audios(client_options, self.agent,
                                                              session_id, self.language_code,
                                                              absolute_path, sheet_name,
                                                              custom_audio_ip, custom_model_type,
                                                              numeric_match,
                                                              data,
                                                              playbook_url,
                                                              page_url,
                                                              parameter_set,
                                                              api_invoke_required)
                        # print(result_df)
                        final_df = pd.concat([final_df, result_df], axis=0)

                    final_df = self.overall_result_check(final_df)
                    overall_pass_fail = list(final_df['Overall'])
                    Pass = overall_pass_fail.count("PASS")
                    Fail = overall_pass_fail.count("FAIL")
                    summary_dict['Type'].append(sheet_name)
                    summary_dict['Total_Record'].append(Pass + Fail)
                    summary_dict['PASS'].append(Pass)
                    summary_dict['FAIL'].append(Fail)
                    summary_dict['PASS(%)'].append(round((Pass / (Pass + Fail)) * 100, 2))


                else:
                    print(data_dict)
                    test_data = data_dict['Utterance']
                    bot_response = data_dict['Response']
                    expected_intents = data_dict['Intent']
                    # expected_entities = data_dict['Entity']
                    expected_entities = data_dict['Entities']
                    levels = data_dict['Levels']
                    test_data_output = []
                    for i in range(len(test_data)):
                        session_id = uuid.uuid4()
                        data = {'test_data': test_data[i], 'bot_response': bot_response[i],
                                'expected_intent': expected_intents[i], 'expected_entities': expected_entities[i],
                                'levels': levels[i], 'expected_threshold': common_threshold_value}

                        result_df,output_dict = self.detect_intent_texts(client_options, self.agent,
                                                             session_id,
                                                             self.language_code,
                                                             data,
                                                             numeric_match,
                                                             playbook_url,
                                                             page_url,
                                                             parameter_set,
                                                             api_invoke_required)
                        test_data_output.append(output_dict)
                        final_df = pd.concat([final_df, result_df], axis=0)
                    print("&&&&&&&&&&&&")
                    print(json.dumps(test_data_output))
                    final_df = self.overall_result_check(final_df)
                    overall_pass_fail = list(final_df['Overall'])
                    Pass = overall_pass_fail.count("PASS")
                    Fail = overall_pass_fail.count("FAIL")
                    summary_dict['Type'].append(sheet_name)
                    summary_dict['Total_Record'].append(Pass + Fail)
                    summary_dict['PASS'].append(Pass)
                    summary_dict['FAIL'].append(Fail)
                    summary_dict['PASS(%)'].append(round((Pass / (Pass + Fail)) * 100, 2))
                print(f"==========Execution completed for {sheet_name}==========")
                final_df.to_excel(writer, engine="openpyxl", sheet_name=sheet_name, index=False)

            summary_dict['Type'].append('Total')
            summary_dict['Total_Record'].append(sum(summary_dict['Total_Record']))
            summary_dict['PASS'].append(sum(summary_dict['PASS']))
            summary_dict['FAIL'].append(sum(summary_dict['FAIL']))
            summary_dict['PASS(%)'].append(
                round((sum(summary_dict['PASS']) / sum(summary_dict['Total_Record'])) * 100, 2))

            summary_df = pd.DataFrame().from_dict(summary_dict)
            summary_df.to_excel(writer, engine='openpyxl', sheet_name='Summary', index=False)
            print('==========Summary creation completed==========')
            writer.close()


        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def get_client_options(self):
        client_options = None
        agent_components = AgentsClient.parse_agent_path(self.agent)
        location_id = agent_components["location"]
        if location_id != "global":
            api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
            client_options = {"api_endpoint": api_endpoint}
        return client_options


    def get_playbooks(self, client_options):
        try:
            playbook_client = PlaybooksClient(client_options=client_options)
            request = ListPlaybooksRequest()
            request.parent = (
                f"projects/{self.PROJECT_ID}/locations/{self.LOCATION_ID}/agents/{self.AGENT_ID}"
            )
            response = playbook_client.list_playbooks(request=request)
            # version_response = playbook_client.list_playbook_versions(request=request)
            json_response = MessageToDict(response._pb)
            playbooks_details = dict()
            for playbook in json_response['playbooks']:
                playbooks_details[playbook['displayName']] = playbook['name']
            return playbooks_details
        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def get_flows(self, client_options):
        try:
            flow_client = FlowsClient(client_options=client_options)
            request = ListFlowsRequest()
            request.parent = (
                f"projects/{self.PROJECT_ID}/locations/{self.LOCATION_ID}/agents/{self.AGENT_ID}"
            )
            request.language_code = "en"
            response = flow_client.list_flows(request=request)
            json_response = MessageToDict(response._pb)
            flows_details = dict()
            for flow in json_response['flows']:
                flows_details[flow['displayName']] = flow['name'].split("/")[-1]
            return flows_details
        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def get_pages(self, client_options, flow_id):
        try:
            page_client = PagesClient(client_options=client_options)
            request = ListPagesRequest()
            request.parent = (
                f"projects/{self.PROJECT_ID}/locations/{self.LOCATION_ID}/agents/{self.AGENT_ID}/flows/{flow_id}"
            )
            request.language_code = "en"
            response = page_client.list_pages(request=request)
            json_response = MessageToDict(response._pb)
            pages_details = dict()
            for page in json_response['pages']:
                pages_details[page['displayName']] = page['name']
            return pages_details
        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def synthesize_text_gcp(self, test_dict, sheet_name):
        """Synthesizes speech from the input string of text."""
        from google.cloud import texttospeech
        import html
        try:
            response_dict = {
                'Utterance': test_dict['Utterance'],
                'Response': test_dict['Response'],
                'Intent': test_dict['Intent'],
                'Entity': test_dict['Entity'],
                'Levels': test_dict['Levels'],
                'Audio_file': []
            }
            client = texttospeech.TextToSpeechClient()
            path = input("Enter the path to create audios : ")
            final_file_path = path + "\\" + sheet_name

            if not os.path.exists(final_file_path):
                os.makedirs(final_file_path)

            # audio creation for level utterances
            for lv in response_dict['Levels']:
                if pd.isna(lv):
                    continue
                else:
                    level = ast.literal_eval(lv)
                    level_key = list(level.keys())
                    for key in level_key:
                        level_text = np.random.choice(level[key])
                        level_input_text = texttospeech.SynthesisInput(text=level_text)
                        voice = texttospeech.VoiceSelectionParams(
                            name="en-US-Standard-B",
                            language_code="en-US"
                        )
                        level_audio_config = texttospeech.AudioConfig(
                            audio_encoding=texttospeech.AudioEncoding.LINEAR16
                        )

                        level_response = client.synthesize_speech(
                            request={"input": level_input_text, "voice": voice, "audio_config": level_audio_config}
                        )
                        full_path = path + "\\" + key
                        if not os.path.exists(full_path):
                            os.makedirs(full_path)
                        level_file_name = full_path + "\\" + level_text + ".wav"
                        # The response's audio_content is binary.
                        with open(level_file_name, "wb") as out:
                            out.write(level_response.audio_content)
                            out.flush()
            pause_time = input("Enter the pause amount in ms (50- 2000) : ")

            """ Parameters to tune the audio 'en-US-Journey-D','en-US-Wavenet-H' """

            VOICE_NAMES = ['en-US-Neural2-D', 'en-US-Neural2-E', 'en-US-Neural2-I',
                           'en-US-Standard-B', 'en-US-Standard-D', 'en-US-Standard-E', 'en-US-Standard-J']

            GENDER = [texttospeech.SsmlVoiceGender.MALE, texttospeech.SsmlVoiceGender.FEMALE]

            RATE = ['+8.00%', '+5.00%', '+4.00%', '+3.00%', '+2.00%', 'medium', '-2.00%', '-3.00%', '+2.00%', '-5.00%',
                    '-8.00%']
            PITCH = ['-2.00%', '-3.00%', '-4.00%', '-5.00%', '-6.00%', '-7.00%', 'medium', '+7.00%', '+6.00%', '+5.00%',
                     '+4.00%', '+3.00%', '+2.00%']
            VOLUME = ['medium', 'loud']
            pause_list = ['<break time="' + pause_time + 'ms" />']

            speech = Speech()
            for i in range(len(response_dict['Utterance'])):
                text = response_dict['Utterance'][i]
                text = html.escape(text)
                temp = list(text)
                temp.insert(0, '<break time="1000ms"/>')
                temp = "".join(temp)
                if ',' in temp:
                    temp = temp.replace(',', np.random.choice(pause_list))

                text = speech.prosody(temp, np.random.choice(RATE), np.random.choice(PITCH), np.random.choice(VOLUME),
                                      True)
                voice_name = np.random.choice(VOICE_NAMES)
                gender = np.random.choice(GENDER)
                # print(text)
                # print(voice_name)
                responseSsml = '<speak>{}</speak>'.format(text)
                print(responseSsml)
                input_text = texttospeech.SynthesisInput(ssml=responseSsml)

                # Note: the voice can also be specified by name.
                # Names of voices can be retrieved with client.list_voices().
                voice = texttospeech.VoiceSelectionParams(
                    name=voice_name,
                    language_code="en-US"
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16
                )

                response = client.synthesize_speech(
                    request={"input": input_text, "voice": voice, "audio_config": audio_config}
                )

                file_name = final_file_path + "\\" + voice_name + "_" + str(i) + ".wav"

                response_dict['Audio_file'].append(voice_name + "_" + str(i) + ".wav")
                # The response's audio_content is binary.
                with open(file_name, "wb") as out:
                    out.write(response.audio_content)
                    out.flush()
            print('Audio creation done...')
            return response_dict, path, final_file_path
        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def synthesize_text_azure(self, test_dict, sheet_name):
        import requests
        try:
            API_KEY = os.getenv("api_key")
            region = os.getenv("region")
            base_url = 'https://{}.tts.speech.microsoft.com/'.format(region)
            url_path = 'cognitiveservices/v1'
            constructed_url = base_url + url_path
            headers = {
                'Ocp-Apim-Subscription-Key': API_KEY,
                'Content-Type': 'application/ssml+xml',
                'X-Microsoft-OutputFormat': 'riff-24khz-16bit-mono-pcm'
            }
            response_dict = {
                'Utterance': test_dict['Utterance'],
                'Response': test_dict['Response'],
                'Intent': test_dict['Intent'],
                'Entity': test_dict['Entity'],
                'Levels': test_dict['Levels'],
                'Audio_file': []
            }

            path = input("Enter the path to create audios : ")
            # audio creation for level utterances
            for lv in response_dict['Levels']:
                if pd.isna(lv):
                    continue
                else:
                    level = ast.literal_eval(lv)
                    level_key = list(level.keys())
                    for key in level_key:
                        level_text = np.random.choice(level[key])
                        voice = 'en-US-EricNeural'
                        full_path = path + "\\" + key
                        if not os.path.exists(full_path):
                            os.makedirs(full_path)

                        level_file_name = full_path + "\\" + level_text + ".wav"
                        if not os.path.exists(level_file_name):
                            request_body = " \
                                           <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'> \
                                               <voice name='{}'> \
                                                   <mstts:silence  type='Leading' value='1000ms'/> \
                                                   {} \
                                               </voice> \
                                           </speak>".format(voice, level_text)
                            level_response = requests.post(constructed_url, headers=headers, data=request_body)
                            # The response's audio_content is binary.
                            if level_response.status_code == 200:
                                with open(level_file_name, 'wb') as audio:
                                    audio.write(level_response.content)

            test_data = list(test_dict['Utterance'])
            pause_amt = input("Enter the pause amount in ms (50 - 2000) : ")
            print('Start converting test utterance to audio file')

            VOICE_NAMES = ['en-US-SaraNeural', 'en-US-MonicaNeural', 'en-US-CoraNeural', 'en-US-NancyNeural',
                           'en-US-JennyNeural', 'en-US-AmberNeural', 'en-US-AriaNeural', 'en-US-ElizabethNeural',
                           'en-US-MichelleNeural', 'en-US-JaneNeural']

            # VOICE_NAMES = ['es-US-PalomaNeural', 'es-US-AlonsoNeural']

            """ List of voices :
            'en-US-SaraNeural', 'en-US-MonicaNeural', 'en-US-CoraNeural', 'en-US-NancyNeural', 'en-US-JennyNeural',
            'en-US-AmberNeural', 'en-US-AriaNeural', 'en-US-ElizabethNeural', 'en-US-MichelleNeural',
            'en-US-JaneNeural' -- Female US
     
     
            'en-US-GuyNeural', 'en-US-JacobNeural', 'en-US-EricNeural',  'en-US-RogerNeural', 'en-US-BrianNeural',
            'en-US-JasonNeural', 'en-US-BrandonNeural', 'en-US-DavisNeural', 'en-US-SteffanNeural', 'en-US-KaiNeural'
            -- Male US
     
     
            'zh-CN-XiaoxiaoNeural', 'zh-CN-XiaoyiNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunyangNeural' -- Chinese voices,
     
     
            'fr-FR-RemyMultilingualNeural', 'fr-FR-VivienneMultilingualNeural' -- French voices,
     
     
            'es-US-PalomaNeural', 'es-US-AlonsoNeural' -- Spanish voices,
     
     
            'en-ZA-LeahNeural', 'en-ZA-LukeNeural' -- African voices,
     
     
            'en-IN-NeerjaNeural', 'en-IN-PrabhatNeural' -- Indian voices
            """

            rate = ['+8.00%', '+5.00%', '+4.00%', '+3.00%', '+2.00%', 'medium', '-2.00%', '-3.00%', '+2.00%', '-5.00%',
                    '-8.00%']
            pitch = ['-2.00%', '-3.00%', '-4.00%', '-5.00%', '-6.00%', '-7.00%', '-9.00%', 'medium', '+9.00%', '+7.00%',
                     '+6.00%', '+5.00%', '+4.00%', '+3.00%', '+2.00%']
            volume = ['medium', 'loud']

            pause_list = ['<break time="' + pause_amt + 'ms" />']
            i = 0
            speech = Speech()
            final_file_path = None
            accent = input("Do you want to run with specific accent (Y/N) : ")
            for j in range(len(test_data)):
                temp = list(test_data[j])
                temp_txt = "".join(temp)
                if ',' in temp:
                    temp = temp_txt.replace(',', np.random.choice(pause_list))
                else:
                    temp = temp_txt
                text = speech.prosody(temp, np.random.choice(rate), np.random.choice(pitch), np.random.choice(volume),
                                      True)
                voice = np.random.choice(VOICE_NAMES)

                if accent.lower() == 'y':
                    text = temp
                    responseSsml = " \
                                   <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'> \
                                       <voice name='{}'> \
                                           <lang xml:lang='en-US'> \
                                               {} \
                                           </lang> \
                                       </voice> \
                                   </speak>".format(voice, text)
                else:
                    responseSsml = " \
                                       <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'> \
                                           <voice name='{}'> \
                                               <mstts:silence  type='Leading' value='1000ms'/> \
                                               {} \
                                           </voice> \
                                       </speak>".format(voice, text)
                print(text)
                body = responseSsml

                response = requests.post(constructed_url, headers=headers, data=body)
                final_file_path = path + "\\" + sheet_name
                if not os.path.exists(final_file_path):
                    os.makedirs(final_file_path)
                file_name = final_file_path + "\\" + voice + "_" + str(i) + ".wav"

                response_dict['Audio_file'].append(voice + "_" + str(i) + ".wav")
                if response.status_code == 200:
                    with open(file_name, 'wb') as audio:
                        audio.write(response.content)
                        # print("\nStatus code: " + str(response.status_code) + "\nYour TTS is ready for playback.\n")
                else:
                    print("\nStatus code: " + str(
                        response.status_code) + "\nSomething went wrong. Check your subscription key and headers.\n")
                    print("Reason: " + str(response.reason) + "\n")
                i += 1
            # print(response_dict)
            print('Audio creation done...')
            return response_dict, path, final_file_path
        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def detect_intent_texts(self, client_options, agent, session_id, language_code, data, numeric_match, playbook, page,
                            parameter_set, api_invoke_required):
        """Returns the result of detect intent with texts as inputs.
        Using the same `session_id` between requests allows continuation
        of the conversation."""
        try:
            text_ip = data['test_data']
            bot_response = data['bot_response']
            expected_intent = data['expected_intent']
            expected_entity = data['expected_entities']
            level = data['levels']
            threshold_value = data['expected_threshold']

            # API call
            # if api_invoke_required.lower() == 'y':
            #     from API_Playbook_call import ApiCall
            #     status_update = ApiCall().main()
            #     if not status_update:
            #         raise Exception("The data has not been set for : ", text_ip)

            session_path = f"{agent}/sessions/{session_id}"

            session_client = SessionsClient(client_options=client_options)

            result_dict = {
                'Utterance': [], 'Expected_Intent': [], 'Expected_Entity': [], 'Expected_Response': [],
                'Expected_Threshold': [], 'Predicted_Intent': [], 'Predicted_Entity': [], 'Bot_Response': [],
                'Level_Utterances': [], 'Confidence': [], 'Session_ID': [], 'Response_ID': [],'Response_Match': [], 'Threshold_Risk': [],
                'Cosine_Similarity_Score': [], 'Cosine_Result': [],
                'CrossEncoder_Contradiction_Score': [], 'CrossEncoder_Entailment_Score': [],
                'CrossEncoder_Neutral_Score': [], 'CrossEncoder_Label': [], 'CrossEncoder_Result': []
            }
            # print(level)
            texts = []
            counter = None
            if not pd.isna(level):
                level = ast.literal_eval(level)
                level_key = list(level.keys())
                start_index = 1
                for key in level_key:
                    seq = int(key.split('_')[1])
                    if seq == start_index:
                        texts.insert(start_index, random.choice(level[key]))
                        start_index += 1
                    else:
                        texts.insert(start_index, random.choice(level[key]))
                        counter = start_index
                        start_index += 2
                    # texts.append(random.choice(level[key]))
                if counter is not None:
                    texts.insert(counter - 1, text_ip)
                else:
                    texts.append(text_ip)
            else:
                texts.append(text_ip)

            level_depth = len(texts)
            # print(texts)
            data_json_response = {}
            for i in range(level_depth):
                text_input = session.TextInput(text=str(texts[i]))
                query_input = session.QueryInput(text=text_input, language_code=language_code)
                query_param = None

                # For Playbook based flow
                if playbook is not None:
                    if parameter_set.lower() == 'y':
                        # Temporary Code for creating parameter
                        from google.protobuf import struct_pb2
                        session_param = "00128096991597426726"
                        parameters = struct_pb2.Struct()
                        parameters.update({"event-WELCOME.CiscoGucid": session_param})
                        query_param = session.QueryParameters(
                            current_playbook=playbook,
                            parameters=parameters
                        )  # Setting Session parameter for skill bot call
                    else:
                        query_param = session.QueryParameters(
                            current_playbook=playbook
                        )
                # For Page based flow
                elif page is not None:
                    if parameter_set.lower() == 'y':
                        # Temporary Code for creating parameter
                        from google.protobuf import struct_pb2
                        session_param = "00128096991597426726"
                        parameters = struct_pb2.Struct()
                        parameters.update({"event-WELCOME.CiscoGucid": session_param})
                        query_param = session.QueryParameters(
                            current_page=page,
                            parameters=parameters
                        )  # Setting Session parameter for skill bot call
                    else:
                        query_param = session.QueryParameters(
                            current_page=page
                        )



                # For Default flow
                else:
                    if parameter_set.lower() == 'y':
                        # Temporary Code for creating parameter
                        from google.protobuf import struct_pb2
                        session_param = "00128096991597426726"
                        parameters = struct_pb2.Struct()
                        parameters.update({"event-WELCOME.CiscoGucid": session_param})
                        query_param = session.QueryParameters(
                            parameters=parameters
                        )
                request = session.DetectIntentRequest(
                    session=session_path, query_input=query_input, query_params=query_param
                )
                response = session_client.detect_intent(request=request)
                json_response = MessageToDict(response._pb)


                if i == level_depth - 1:
                    print("Printing JSON response")
                    print(json.dumps(json_response))
                    data_json_response = json_response
                    print("================")
                    result_dict['Utterance'].append(texts[i] if counter is None else texts[counter - 1])
                    result_dict['Expected_Intent'].append(expected_intent if not pd.isna(expected_intent) else ' ')
                    result_dict['Expected_Entity'].append(expected_entity if not pd.isna(expected_entity) else ' ')
                    result_dict['Expected_Response'].append(bot_response if not pd.isna(bot_response) else ' ')
                    result_dict['Expected_Threshold'].append(threshold_value)
                    result_dict['Level_Utterances'].append(level if not pd.isna(level) else ' ')
                    result_dict['Session_ID'].append(json_response['queryResult']['diagnosticInfo']['Session Id'])
                    result_dict['Response_ID'].append(json_response['queryResult']['diagnosticInfo']['Response Id'])

                    # Threshold risk check section
                    intent_confidence = response.query_result.intent_detection_confidence * 100
                    result_dict['Confidence'].append(intent_confidence)
                    result_dict['Threshold_Risk'].append("PASS" if intent_confidence >= threshold_value else "FAIL")

                    # Response match section - cosine_similarity and cross_encoder
                    response_messages = [
                        " ".join(msg.text.text) for msg in response.query_result.response_messages
                    ]
                    response_messages = ' '.join(response_messages)

                    result_dict['Bot_Response'].append(response_messages)
                    if not pd.isna(bot_response) and len(response_messages.strip()) > 0:
                        cosine_score, cosine_result = self.response_match_cosine(response_messages, bot_response)

                        ce_scores = self.compute_metric(response_messages, bot_response)

                        result_dict['Response_Match'].append(cosine_result)
                        result_dict['Cosine_Similarity_Score'].append(round(cosine_score, 4))
                        result_dict['Cosine_Result'].append(cosine_result)

                        result_dict['CrossEncoder_Contradiction_Score'].append(ce_scores['contradiction_score'])
                        result_dict['CrossEncoder_Entailment_Score'].append(ce_scores['entailment_score'])
                        result_dict['CrossEncoder_Neutral_Score'].append(ce_scores['neutral_score'])
                        result_dict['CrossEncoder_Label'].append(ce_scores['label'])
                        result_dict['CrossEncoder_Result'].append(ce_scores['result'])

                    else:
                        result_dict['Response_Match'].append(" ")
                        result_dict['Cosine_Similarity_Score'].append("")
                        result_dict['Cosine_Result'].append("")
                        result_dict['CrossEncoder_Contradiction_Score'].append("")
                        result_dict['CrossEncoder_Entailment_Score'].append("")
                        result_dict['CrossEncoder_Neutral_Score'].append("")
                        result_dict['CrossEncoder_Label'].append("")
                        result_dict['CrossEncoder_Result'].append("")

                    # Intent match section
                    if "intent" in json_response['queryResult'].keys():
                        predicted_intent = json_response['queryResult']['intent']['displayName']
                        result_dict['Predicted_Intent'].append(predicted_intent)
                        if not pd.isna(expected_intent):
                            result_dict['Intent_Match'] = []
                            result_dict['Intent_Match'].append(
                                "PASS" if expected_intent.replace(" ",
                                                                  "") == predicted_intent.replace(" ",
                                                                                                  "") else "FAIL")
                    else:
                        result_dict['Predicted_Intent'].append(" ")

                    # Entity match section
                    # print(repr(expected_entity))
                    if not pd.isna(expected_entity):
                        # print(type(expected_entity))
                        # print(expected_entity)
                        # print("-----------------")
                        entity = ast.literal_eval(str(expected_entity))
                        predicted_entity = json_response['queryResult']
                        result_dict['Entity_Match'] = []
                        entity_status, predicted_entity_response = self.entity_validation(entity, predicted_entity,
                                                                                          numeric_match)
                        result_dict['Predicted_Entity'].append(predicted_entity_response)
                        result_dict['Entity_Match'].append(entity_status)
                    else:
                        result_dict['Predicted_Entity'].append(' ')

            # print(result_dict)
            print("\n===== Checking result_dict lengths =====")
            for key, val in result_dict.items():
                print(f"{key}: {len(val)}")
            print("========================================\n")
            result_df = pd.DataFrame().from_dict(result_dict)
            score_df = result_df[[
                'Utterance',
                'Bot_Response',
                'Cosine_Similarity_Score',
                'Cosine_Result',
                'CrossEncoder_Contradiction_Score',
                'CrossEncoder_Entailment_Score',
                'CrossEncoder_Neutral_Score',
                'CrossEncoder_Label'
            ]]

            csv_path = os.path.abspath("utterance_scores_only.csv")
            score_df.to_csv(csv_path, index=False)

            print(f"Score-only CSV saved at: {csv_path}")
            return result_df, score_df


        except Exception as e:
            import sys
            print("Error in : " + str(e) + " " + str(sys.exc_info()[-1].tb_lineno))


    def detect_intent_audios(self, client_options, agent, session_id, language_code,
                             path, sheet_name, custom_stt, custom_model_type, numeric_match, data, playbook, page,
                             parameter_set, api_invoke_required):
        try:
            text_ip = data['test_data']
            bot_response = data['bot_response']
            expected_intent = data['expected_intent']
            expected_entity = data['expected_entity']
            level = data['levels']
            audio_file = data['audio_file']
            threshold_value = data['expected_threshold']

            # API call
            # if api_invoke_required.lower() == 'y':
            #     from API_Playbook_call import ApiCall
            #     status_update = ApiCall().main()
            #     if not status_update:
            #         raise Exception("The data has not been set for : ", text_ip)

            session_path = f"{agent}/sessions/{session_id}"
            session_client = SessionsClient(client_options=client_options)

            result_dict = {
                'Utterance': [], 'Audio': [], 'Audio_Transcript': [], 'Expected_Intent': [], 'Expected_Entity': [],
                'Expected_Response': [], 'Expected_Threshold': [], 'Predicted_Intent': [], 'Predicted_Entity': [],
                'Bot_Response': [],
                'Level_Utterances': [], 'Confidence': [], 'Session_ID': [], 'Response_ID': [], 'Threshold_Risk': []
            }

            audios = []
            counter = None
            audio_ip = glob.glob(f"{path}\\{sheet_name}\\{audio_file}")
            if not pd.isna(level):
                level = ast.literal_eval(level)
                level_key = list(level.keys())
                start_index = 1
                for key in level_key:
                    specific_audio = glob.glob(f"{path}\\{key}\\{level[key][0]}.wav")
                    # print(specific_audio)
                    seq = int(key.split('_')[1])
                    if seq == start_index:
                        audios.insert(start_index, specific_audio[0])
                        start_index += 1
                    else:
                        audios.insert(start_index, specific_audio[0])
                        counter = start_index
                        start_index += 2
                    # audios.append(list_of_audios[0])
                if counter is not None:
                    audios.insert(counter - 1, audio_ip[0])
                else:
                    audios.append(audio_ip[0])
            else:
                audios.append(audio_ip[0])
            level_depth = len(audios)
            audio_response = list()

            query_param = None

            # Playbook based
            if playbook is not None:
                if parameter_set.lower() == 'y':
                    # Temporary Code for creating parameter
                    from google.protobuf import struct_pb2
                    session_param = "00128096991597426726"
                    parameters = struct_pb2.Struct()
                    parameters.update({"event-WELCOME.CiscoGucid": session_param})
                    query_param = session.QueryParameters(
                        current_playbook=playbook,
                        parameters=parameters
                    )  # Setting Session parameter for skill bot call
                else:
                    query_param = session.QueryParameters(
                        current_playbook=playbook
                    )


            # Page based
            elif page is not None:
                if parameter_set.lower() == 'y':
                    # Temporary Code for creating parameter
                    from google.protobuf import struct_pb2
                    session_param = "00128096991597426726"
                    parameters = struct_pb2.Struct()
                    parameters.update({"event-WELCOME.CiscoGucid": session_param})
                    query_param = session.QueryParameters(
                        current_page=page,
                        parameters=parameters
                    )  # Setting Session parameter for skill bot call
                else:
                    query_param = session.QueryParameters(
                        current_page=page
                    )
            # Default flow
            else:
                if parameter_set.lower() == 'y':
                    # Temporary Code for creating parameter
                    from google.protobuf import struct_pb2
                    session_param = "00128096991597426726"
                    parameters = struct_pb2.Struct()
                    parameters.update({"event-WELCOME.CiscoGucid": session_param})
                    query_param = session.QueryParameters(
                        parameters=parameters
                    )

            for i in range(level_depth):
                audio = audios[i]
                if custom_stt.lower() == 'y' and audio == audio_ip[0]:
                    # print(audio)
                    cog_key = os.getenv('api_key')
                    cog_region = os.getenv('region')
                    # Configure speech service
                    speech_config = speech_sdk.SpeechConfig(cog_key, cog_region)
                    if int(custom_model_type) == 1:
                        cog_key = ""
                        cog_region = ""
                        model_id = ""
                        # Configure speech service
                        speech_config = speech_sdk.SpeechConfig(cog_key, cog_region)
                        speech_config.endpoint_id = ""

                        speech_config.set_service_property("postprocessing", model_id,
                                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)
                        speech_config.output_format = speech_sdk.OutputFormat.Detailed
                        speech_config.set_property(speech_sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, '1000')
                    elif int(custom_model_type) == 2:
                        cog_key = ""
                        cog_region = ""
                        model_id = ""
                        # Configure speech service
                        speech_config = speech_sdk.SpeechConfig(cog_key, cog_region)
                        speech_config.endpoint_id = ""
                        speech_config.enable_dictation()
                        speech_config.set_service_property("postprocessing", model_id,
                                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)
                        speech_config.set_service_property('punctuation', 'explicit',
                                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)
                        speech_config.output_format = speech_sdk.OutputFormat.Detailed
                        speech_config.set_property(speech_sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, '1000')
                    elif int(custom_model_type) == 3:

                        cog_key = ""
                        cog_region = ""
                        model_id = ""
                        # Configure speech service
                        speech_config = speech_sdk.SpeechConfig(cog_key, cog_region)
                        speech_config.endpoint_id = ""
                        speech_config.set_service_property("postprocessing", model_id,
                                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)
                        speech_config.output_format = speech_sdk.OutputFormat.Detailed
                        speech_config.set_property(speech_sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, '1000')
                    elif int(custom_model_type) == 4:
                        cog_key = ""
                        cog_region = ""
                        model_id = ""
                        # Configure speech service
                        speech_config = speech_sdk.SpeechConfig(cog_key, cog_region)
                        speech_config.endpoint_id = ""

                        speech_config.set_service_property("postprocessing", model_id,
                                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)
                        speech_config.output_format = speech_sdk.OutputFormat.Detailed
                        speech_config.set_property(speech_sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, '1000')
                    elif int(custom_model_type) == 5:
                        cog_key = ""
                        cog_region = ""
                        model_id = ""
                        # Configure speech service
                        speech_config = speech_sdk.SpeechConfig(cog_key, cog_region)
                        speech_config.endpoint_id = ""

                        speech_config.set_service_property("postprocessing", model_id,
                                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)
                        speech_config.output_format = speech_sdk.OutputFormat.Detailed
                        speech_config.set_property(speech_sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, '1000')
                    elif int(custom_model_type) == 6:
                        cog_key = ""
                        cog_region = ""
                        model_id = ""
                        # Configure speech service
                        speech_config = speech_sdk.SpeechConfig(cog_key, cog_region)
                        speech_config.endpoint_id = ""

                        speech_config.set_service_property("postprocessing", model_id,
                                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)
                        speech_config.output_format = speech_sdk.OutputFormat.Detailed
                        speech_config.set_property(speech_sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, '1000')
                    elif int(custom_model_type) == 7:
                        cog_key = ""
                        cog_region = ""
                        model_id = ""
                        # Configure speech service
                        speech_config = speech_sdk.SpeechConfig(cog_key, cog_region)
                        speech_config.endpoint_id = ""

                        speech_config.set_service_property("postprocessing", model_id,
                                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)
                        speech_config.set_property(speech_sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, '1000')
                        speech_config.output_format = speech_sdk.OutputFormat.Detailed

                    stt_output = self.transcribe_command(speech_config, audio)
                    text_input = session.TextInput(text=stt_output)
                    query_input = session.QueryInput(text=text_input, language_code=language_code)

                    request = session.DetectIntentRequest(
                        session=session_path, query_input=query_input, query_params=query_param
                    )
                    response = session_client.detect_intent(request=request)
                    json_response = MessageToDict(response._pb)
                    # print(json_response)
                    audio_response.append(stt_output)
                else:
                    input_audio_config = audio_config.InputAudioConfig(
                        audio_encoding=audio_config.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
                        sample_rate_hertz=24000
                    )
                    with open(audio, "rb") as audio_file:
                        input_audio = audio_file.read()
                    audio_input = session.AudioInput(config=input_audio_config, audio=input_audio)
                    query_input = session.QueryInput(audio=audio_input, language_code=language_code)

                    request = session.DetectIntentRequest(
                        session=session_path, query_input=query_input, query_params=query_param
                    )
                    response = session_client.detect_intent(request=request)
                    json_response = MessageToDict(response._pb)
                    # print(response.query_result.transcript)
                    audio_response.append(response.query_result.transcript)
                if i == level_depth - 1:
                    # print(json_response)
                    # print(text_ip)
                    # print(response.query_result.transcript)
                    # print(audios)
                    result_dict['Utterance'].append(text_ip)
                    result_dict['Audio'].append(
                        audio.split("\\")[-1] if counter is None else audios[counter - 1].split("\\")[-1])
                    result_dict['Audio_Transcript'].append(
                        audio_response[i] if counter is None else audio_response[counter - 1])
                    result_dict['Expected_Intent'].append(expected_intent if not pd.isna(expected_intent) else ' ')
                    result_dict['Expected_Entity'].append(expected_entity if not pd.isna(expected_entity) else ' ')
                    result_dict['Expected_Response'].append(bot_response if not pd.isna(bot_response) else ' ')
                    result_dict['Expected_Threshold'].append(threshold_value)
                    result_dict['Level_Utterances'].append(level if not pd.isna(level) else ' ')
                    result_dict['Session_ID'].append(json_response['queryResult']['diagnosticInfo']['Session Id'])
                    result_dict['Response_ID'].append(json_response['queryResult']['diagnosticInfo']['Response Id'])

                    # Threshold risk check section
                    intent_confidence = response.query_result.intent_detection_confidence * 100
                    result_dict['Confidence'].append(intent_confidence)
                    result_dict['Threshold_Risk'].append("PASS" if intent_confidence >= threshold_value else "FAIL")

                    # Response match section
                    result_dict['Response_Match'] = []
                    response_messages = [
                        " ".join(msg.text.text) for msg in response.query_result.response_messages
                    ]
                    response_messages = ' '.join(response_messages)
                    result_dict['Bot_Response'].append(response_messages)
                    if not pd.isna(bot_response) and len(response_messages) > 0:
                        if self.response_match_cosine(response_messages, bot_response):
                            result_dict['Response_Match'].append("PASS")
                        else:
                            result_dict['Response_Match'].append("FAIL")
                    else:
                        result_dict['Response_Match'].append(" ")

                    # Intent match section
                    if "intent" in json_response['queryResult'].keys():
                        predicted_intent = json_response['queryResult']['intent']['displayName']
                        result_dict['Predicted_Intent'].append(predicted_intent)
                        if not pd.isna(expected_intent):
                            result_dict['Intent_Match'] = []
                            result_dict['Intent_Match'].append(
                                "PASS" if expected_intent.replace(" ", "") == predicted_intent.replace(" ",
                                                                                                       "") else "FAIL")
                    else:
                        result_dict['Predicted_Intent'].append(" ")

                    # Entity match section
                    if not pd.isna(expected_entity):
                        entity = ast.literal_eval(expected_entity)
                        predicted_entity = json_response['queryResult']
                        result_dict['Entity_Match'] = []
                        entity_status, predicted_entity_response = self.entity_validation(entity,
                                                                                          predicted_entity,
                                                                                          numeric_match)
                        result_dict['Predicted_Entity'].append(predicted_entity_response)
                        result_dict['Entity_Match'].append(entity_status)
                    else:
                        result_dict['Predicted_Entity'].append(' ')
            print(result_dict)
            result_df = pd.DataFrame().from_dict(result_dict)
            return result_df


        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def entity_validation(self, expected_entity, predicted_entity, numeric_check):
        try:
            entity_pass_fail = ' '
            predicted_entity_response = ' '
            if 'entity' in expected_entity.keys():
                if "match" in predicted_entity.keys() and "parameters" in \
                        predicted_entity['match'].keys():
                    pred_ent = predicted_entity['match']["parameters"]
                    predicted_entity_response = pred_ent
                    all_entities = expected_entity['entity']

                    for key in all_entities.keys():
                        exp_entity_key = key
                        exp_entity_val = all_entities[key]

                        # Key match
                        if len(exp_entity_val) == 0:
                            if exp_entity_key in pred_ent.keys():
                                entity_pass_fail = "PASS"
                            else:
                                entity_pass_fail = "FAIL"


                        # Dict type entity value match
                        elif isinstance(exp_entity_val[0], dict):
                            if exp_entity_key in pred_ent.keys():
                                pred_dos_val_dict = pred_ent[exp_entity_key]
                                inner_dict = exp_entity_val[0]
                                for dos_key in inner_dict.keys():
                                    dos_value = inner_dict[dos_key]
                                    if dos_key in pred_dos_val_dict.keys():
                                        if dos_value == pred_dos_val_dict[dos_key]:
                                            entity_pass_fail = "PASS"
                                        else:
                                            entity_pass_fail = "FAIL"
                                    else:
                                        entity_pass_fail = "PASS"
                            else:
                                entity_pass_fail = "FAIL"


                        # key-value match
                        else:
                            # In case it's a numeric match
                            if numeric_check == 'y':
                                if exp_entity_key in pred_ent.keys():
                                    def value_format():
                                        return lambda x: re.sub(" ", "", x)

                                    pred_ent_value = list(map(value_format(), pred_ent.values()))
                                    entity_pass_fail = 'PASS' if exp_entity_val[0] in pred_ent_value else 'FAIL'


                            else:
                                if exp_entity_key in pred_ent.keys():
                                    entity_pass_fail = 'PASS' if exp_entity_val[0] in pred_ent.values() else 'FAIL'
                                else:
                                    entity_pass_fail = "FAIL"
            elif "parameter" in expected_entity.keys():
                if "parameters" in predicted_entity.keys():
                    predicted_entity_response = predicted_entity['parameters']
                    all_entities = expected_entity['parameter']
                    # For multiple key-value pair in expected entity column
                    for key in all_entities.keys():
                        exp_entity_key = key
                        exp_entity_val = all_entities[key]

                        def check_instance(val):
                            if isinstance(val, dict):
                                return True

                        dict_type_parameter_values = list(
                            filter(check_instance, predicted_entity['parameters'].values()))

                        if exp_entity_key in predicted_entity['parameters'].keys():
                            if len(exp_entity_val) > 0:
                                pred_entity_val = predicted_entity['parameters'][exp_entity_key]
                                if exp_entity_val[0] != pred_entity_val:
                                    entity_pass_fail = "FAIL"
                                    break
                                else:
                                    entity_pass_fail = "PASS"
                            else:
                                entity_pass_fail = "PASS"


                        elif len(dict_type_parameter_values) > 0:
                            # print(dict_type_parameter_values)
                            for val_dict in dict_type_parameter_values:
                                if exp_entity_key in val_dict.keys():
                                    pred_entity_val = val_dict[exp_entity_key]
                                    if exp_entity_val[0] != pred_entity_val:
                                        entity_pass_fail = "FAIL"
                                        break
                                    else:
                                        entity_pass_fail = "PASS"
                                        break
                                else:
                                    entity_pass_fail = "FAIL"
                                    break
                        else:
                            entity_pass_fail = "FAIL"
                            break
            print("Entities->", predicted_entity_response)
            return entity_pass_fail, predicted_entity_response


        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def transcribe_command(self, speech_config, audio):
        speech_config.set_service_property('punctuation', 'explicit',
                                           speech_sdk.ServicePropertyChannel.UriQueryParameter)

        audio_configuration = speech_sdk.AudioConfig(filename=audio)
        speech_recognizer = speech_sdk.SpeechRecognizer(speech_config, audio_configuration)
        data = None
        # Process speech input
        speech = speech_recognizer.recognize_once()
        if speech.reason == speech_sdk.ResultReason.RecognizedSpeech:
            data = speech.text
            # print(data)
        else:
            print(speech.reason)
            if speech.reason == speech_sdk.ResultReason.Canceled:
                cancellation = speech.cancellation_details
                print(cancellation.reason)
                print(cancellation.error_details)
            elif speech.reason == speech_sdk.ResultReason.NoMatch:
                data = speech.reason

        # Return the data
        return data


    def overall_result_check(self, data_df):
        try:
            result_dict = data_df.to_dict(orient='list')
            result_dict['Overall'] = []
            column_list = ['Response_Match', 'Entity_Match', 'Intent_Match', 'Threshold_Risk']
            for i in range(len(result_dict['Utterance'])):
                """ Overall result check """
                pass_fail_list = list(
                    (result_dict[column][i] for column in column_list if column in result_dict.keys()))
                if 'FAIL' in pass_fail_list:
                    result_dict['Overall'].append("FAIL")
                else:
                    result_dict['Overall'].append("PASS")
            result_df = pd.DataFrame().from_dict(result_dict)
            return result_df
        except Exception as e:
            import sys
            print("Error in : " + str(e) + str(sys.exc_info()[-1].tb_lineno))


    def response_match_cosine(self, sentence1, sentence2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import re
        processed_sentence_list = list((re.sub(r"<[^>]+>", "", sentence) for sentence in [sentence1, sentence2]))
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(processed_sentence_list)
        similarity_matrix = cosine_similarity(vectors[0], vectors[1])
        similarity_score = float(similarity_matrix[0][0])
        result = "PASS" if similarity_score > 0.6 else "FAIL"
        return similarity_score, result

    def compute_metric(self, ground_truth: str, inference: str) -> dict:
        scores = self.nli_model.predict([(ground_truth, inference)], apply_softmax=True)  # shape: (1, 3)

        if scores is None or len(scores) == 0:
            return {
                'contradiction_score': "",
                'entailment_score': "",
                'neutral_score': "",
                'label': "",
                'result': ""
            }

        score_arr = scores[0]  # Now it's [contradiction, entailment, neutral]
        label_index = score_arr.argmax()
        label = ['contradiction', 'entailment', 'neutral'][label_index]

        return {
            'contradiction_score': round(float(score_arr[0]), 4),
            'entailment_score': round(float(score_arr[1]), 4),
            'neutral_score': round(float(score_arr[2]), 4),
            'label': label,
            'result': "PASS" if label == "entailment" or float(score_arr[1]) > 0.6 else "FAIL"
        }


if __name__ == "__main__":
    obj_df = DialogflowCXAssurance()
obj_df.main()
