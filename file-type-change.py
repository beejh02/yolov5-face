import ffmpeg

ffmpeg.input('./myimages/youtube_video.mp4') \
      .output('output.avi', vcodec='ffv1') \
      .run()
