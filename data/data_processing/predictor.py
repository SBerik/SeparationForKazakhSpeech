from functional import mapper_from_flac_to_df

def get_VAD(audio_name, base_path, sr = 16000, model = False):
    if not model:
        csv_i = mapper_from_flac_to_df (audio_name, base_path)
        return csv_i
    else:
        #ToDo
        pass  