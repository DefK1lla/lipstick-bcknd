import { pipeline, env } from '@xenova/transformers';
import fs from 'fs'
import express  from 'express';
import cors from 'cors';
import path from 'path';
import { tmpdir } from 'os';
;

env.allowLocalModels = false;

const parseFace = async (url) => {
 try {
  const classifier = await pipeline('image-segmentation', 'Xenova/face-parsing');
  return await classifier(url);
 } catch(e) {
  console.log(e.stack)
 }
}


function formatData(data) {
  const map = {
    lLip: 'l_lip',
    uLip: 'u_lip',
  }
  const NEED_KEY = [map.lLip, map.uLip];
  const temp1 = data.filter((it) => NEED_KEY.includes(it.label));
  const temp2 = temp1.reduce((sum, it) => ({ ...sum, [it.label]: it.mask.data }), {})
  const { l_lip, u_lip } = temp2;
  const { width, height } = temp1[0].mask;
  const lipData = [];

  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      const currentIndex = j * width + i;

      if (l_lip?.[currentIndex] || u_lip?.[currentIndex]) {
        lipData.push(currentIndex)
      }
    }
  }

  return {
    lipData
  }
}

const app = express();

app.use(cors({
  origin: '*'
}));
app.use(express.json({limit: '50mb'}));

app.listen(4000, () => {
  console.log('Server started', 4000);
});

app.post('/api', (req, res) => {
  const imgBase64 = req.body.imgBase64;
  const formatedBase64 = imgBase64.split(';base64,').pop();

  fs.writeFile(path.join(tmpdir(), 'image.png'), formatedBase64, { encoding: 'base64' }, function(err) {
    if (err) { 
      console.error(err);
      return res.status(500).send('Unknown Error').end();
    }
    parseFace('./tmp/image.png')
      .then(formatData)
      .then(data => res.send(data));
  });
});