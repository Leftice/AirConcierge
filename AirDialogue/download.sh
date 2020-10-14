set -e # stop script when there is an error

# download airdialogue data
## WGET ## refer from https://www.matthuisman.nz/2019/01/download-google-drive-files-wget-curl.html

export fileid=1rtKhWK4Ca-VBi2gRqEpjuJma_DMjP6W_
export filename=data.tar.gz

wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

tar -xzvf data.tar.gz
