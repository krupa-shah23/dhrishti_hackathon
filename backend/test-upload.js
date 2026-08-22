const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

async function testUpload() {
  try {
    fs.writeFileSync('test.mp4', 'dummy video content');
    const form = new FormData();
    form.append('video', fs.createReadStream('test.mp4'));
    
    const res = await axios.post('http://localhost:5000/api/videos/upload', form, {
      headers: form.getHeaders()
    });
    console.log('Success:', res.data);
  } catch (err) {
    console.error('Upload failed:', err.message);
    if (err.response) console.error(err.response.data);
  }
}

testUpload();
