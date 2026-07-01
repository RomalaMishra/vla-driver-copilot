# docs

Once `python -m demo.make_demo_video` has produced `outputs/demo_reel.mp4`,
convert a representative slice to a gif and drop it here as `demo.gif`
(referenced by the main README):

```bash
ffmpeg -i outputs/demo_reel.mp4 -vf "fps=8,scale=640:-1" docs/demo.gif
```
