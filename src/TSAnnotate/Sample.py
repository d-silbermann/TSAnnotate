from torchcodec.decoders import VideoDecoder
from torchvision.transforms.v2.functional import to_pil_image


def sample_video(video_path, sample_rate=1):
    video = VideoDecoder(video_path)
    num_frames = int(video.metadata.duration_seconds * sample_rate) + 1
    # check sample rate
    fps = video.metadata.average_fps
    assert sample_rate <= fps, ("Sample rate (" + str(sample_rate) + ") should not be larger than video FPS ("
                                + str(fps) + ")")
    assert sample_rate > 0, "Sample rate should be larger than 0"
    i = 1
    print()
    # extract frames
    for frame in video[::int(round(fps / sample_rate, 0))]:
        print(f"\rExtracting frame {i}/{num_frames}", end="", flush=True)
        i += 1
        yield to_pil_image(frame)
