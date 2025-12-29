#!/bin/bash

folder="${1:-.}"

find "$folder" -maxdepth 1 -type f \( \
    -iname "*.mp4" -o \
    -iname "*.mov" -o \
    -iname "*.avi" -o \
    -iname "*.mkv" -o \
    -iname "*.webm" \
\) | sort | while read -r video; do

    while true; do
        echo "Playing: $(basename "$video")"
        echo "Controls: SPACE=pause, q=next, d=delete, x=quit, 1-9=trim last N seconds"

        rm -f /tmp/mpv_action

        mpv --really-quiet --keep-open=yes --vf=scale=640:640 --input-conf=/dev/stdin "$video" <<'EOF'
d run "/bin/bash" "-c" "echo DELETE > /tmp/mpv_action"; quit
D run "/bin/bash" "-c" "echo DELETE > /tmp/mpv_action"; quit
q run "/bin/bash" "-c" "echo NEXT > /tmp/mpv_action"; quit
x run "/bin/bash" "-c" "echo QUIT > /tmp/mpv_action"; quit
X run "/bin/bash" "-c" "echo QUIT > /tmp/mpv_action"; quit
1 run "/bin/bash" "-c" "echo TRIM 1 > /tmp/mpv_action"; quit
2 run "/bin/bash" "-c" "echo TRIM 2 > /tmp/mpv_action"; quit
3 run "/bin/bash" "-c" "echo TRIM 3 > /tmp/mpv_action"; quit
4 run "/bin/bash" "-c" "echo TRIM 4 > /tmp/mpv_action"; quit
5 run "/bin/bash" "-c" "echo TRIM 5 > /tmp/mpv_action"; quit
6 run "/bin/bash" "-c" "echo TRIM 6 > /tmp/mpv_action"; quit
7 run "/bin/bash" "-c" "echo TRIM 7 > /tmp/mpv_action"; quit
8 run "/bin/bash" "-c" "echo TRIM 8 > /tmp/mpv_action"; quit
9 run "/bin/bash" "-c" "echo TRIM 9 > /tmp/mpv_action"; quit
EOF

        if [ -f /tmp/mpv_action ]; then
            action=$(cat /tmp/mpv_action)
            rm /tmp/mpv_action

            if [ "$action" = "DELETE" ]; then
                rm "$video"
                echo "Deleted: $(basename "$video")"
                break
            elif [ "$action" = "NEXT" ]; then
                break
            elif [ "$action" = "QUIT" ]; then
                echo "Quitting..."
                exit 0
            elif [[ "$action" == TRIM* ]]; then
                seconds="${action#TRIM }"
                duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$video")
                new_duration=$(echo "$duration - $seconds" | bc)
                if (( $(echo "$new_duration > 0" | bc -l) )); then
                    tmp_file="${video%.*}_trimmed.${video##*.}"
                    ffmpeg -y -v error -i "$video" -t "$new_duration" -c copy "$tmp_file"
                    mv "$tmp_file" "$video"
                    echo "Trimmed $seconds seconds from end"
                else
                    echo "Cannot trim: video too short"
                fi
            fi
        fi
    done

    echo ""
done

echo "Done reviewing videos."
