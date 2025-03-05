#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Anonymize the text and media associated with a given set of ELAN transcripts.
#
# Usage:
#
#   anonymize_doc.py EXAMPLE/srs-DCT-20181201-02-02.eaf
#

#
# TODO: For now, we use ffmpeg(1) to filter sensitive portions of video and
#       audio, applying a full-frame blur in the case of video and silence in
#       the case of audio.  This works, but there are many ways in which this
#       could be better:
#
#           1. Rather than blur the entire video by default, it would be much
#              nicer if we could simply blur speakers' faces during sensitive
#              portions.
#
#              We could split up our videos into a sequence of non-sensitive
#              and sensitive clips, apply 'deface' to the sensitive clips,
#              then stitch them all back together using ffmpeg(1):
#
#                   https://github.com/ORB-HD/deface
#
#              (We would need to make sure that this doesn't affect the total
#              duration of the video (e.g., by losing frames at the cut points,
#              if ffmpeg(1) accidentally does that), which could lead to the
#              audio being out of sync, but if that doesn't happen, then this
#              approach should be fine -- and could potentially save us from
#              having to re-encode the entire video, assuming 'deface' can
#              reproduce the original encoding.)
#
#           2. We could follow the lead of some other projects and filter
#              the audio in some fancier way -- cf. Daniel Hirst's replacing
#              speech with a hum that follows the same prosodic contours, or
#              SBCSAE's FIR low-pass filter approach.  For now, silence is
#              probably OK, but this might be worth looking into, even if
#              it requires some calls to Praat or the same kind of splitting-
#              then-stitching approach as with 'deface' above to apply
#              ffmpeg(1) filters that don't offer "enable" (which happens to
#              include their 'firequalizer' filter, sadly).
#

import argparse
import os
import os.path
import re
import shutil
import subprocess
import sys
import tempfile

import pympi


# How many milliseconds of context to include before and after each clip
# produced to review the results of anonymization?
REVIEW_CONTEXT = 1000


# This nonsense is needed to change the value of an existing annotation using
# pympi-ling. 
def set_ref_annotation(transcript, tier, start_ms, end_ms, new_value):
    parent_tier = transcript.tiers[tier][2]['PARENT_REF']

    for (ref_ann_id, (parent_ann_id, ref_ann_value, ref_ann_prev, \
                      ref_ann_svg)) in transcript.tiers[tier][1].items():
        # Grab the (in this case, time-aligned) parent tier of this
        # annotation, since we'll need its start and end times for our
        # search.
        parent_start_ts, parent_end_ts, parent_ann_value, parent_refvalue = \
            transcript.tiers[parent_tier][0][parent_ann_id]

        if transcript.timeslots[parent_start_ts] == start_ms and \
           transcript.timeslots[parent_end_ts] == end_ms:
            transcript.tiers[tier][1][ref_ann_id] = (parent_ann_id, \
                new_value, ref_ann_prev, ref_ann_svg)
            break

def set_aligned_annotation(transcript, tier, start_ms, end_ms, new_value):
    for (annotation_id, (start_ts, end_ts, value, refvalue)) in \
        transcript.tiers[tier][0].items():
        if transcript.timeslots[start_ts] == start_ms and \
           transcript.timeslots[end_ts] == end_ms:
            transcript.tiers[tier][0][annotation_id] = (start_ts, end_ts, \
                new_value, refvalue)
            break

# Replace any text that is marked up as needing to be anonymized with
# appropriate placeholders.
def anonymize_text(s):
    # '[anon type="name"]...[/anon]' or '[name]...[/name]' --> '(name)'
    s = re.sub('\[anon type="name"\](.*?)\[\/anon\]', '(NAME)', s)
    s = re.sub('\[name\](.*?)\[\/name\]', '(NAME)', s)

    # '[anon type="language|topic|sensitive"]...[/anon]' or 
    # '[sensitive]...[/sensitive]' --> '(sensitive)'
    s = re.sub('\[anon type=".*?"\](.*?)\[\/anon\]', '(SENSITIVE)', s)
    s = re.sub('\[sensitive\](.*?)\[\/sensitive\]', '(SENSITIVE)', s)

    return s

# Create an anonymized version of the given audio file, replacing each of the
# specified segments in it with silence and storing the result in the given
# output directory with the given suffix appended between the file's basename
# and its extension (e.g., "/path/to/output_dir/BASENAME-OUTPUT_SUFFIX.wav").
def anonymize_audio(input_audio_fname, segments_to_anonymize, output_dir, \
                    output_suffix):
    # Construct a string of filter commands for ffmpeg(1) that sets the volume
    # to zero (= silences the audio) at the intervals specified in the tuples
    # given in 'anns_to_anonymize'.  This should look like (all on one line):
    #
    #   volume=enable='between(t\,216.670\,219.193)':volume=0, 
    #   volume=enable='between(t\,351.430\,352.264)':volume=0,
    #       (etc.)
    #
    # These definitions can get longer than the command line can handle, so we
    # write them to a named temporary file that ffmpeg(1) can then read using
    # its 'filter_complex_script' option.
    volume_filter = ', '.join(\
        ["volume=enable='between(t\,%.3f\,%.3f)':volume=0" % \
         (start_ms / 1000.0, end_ms / 1000.0) for \
         (start_ms, end_ms, _) in segments_to_anonymize])

    volume_filter_f = tempfile.NamedTemporaryFile(mode = 'w', \
        delete = False, suffix = '.txt')
    volume_filter_f.write(volume_filter)
    volume_filter_f.close()

    # Store the anonymized audio in 'output_dir' under the name '${BASENAME}-
    # ${OUTPUT_SUFFIX}.wav" by default.
    output_audio_fname = os.path.join(output_dir, \
        os.path.splitext(os.path.basename(input_audio_fname))[0] + \
        output_suffix + '.wav')

    # Process the audio using ffmpeg(1).
    subprocess.call(['ffmpeg', \
        '-y', \
        '-v', '0', \
        '-i', input_audio_fname, \
        '-c:a', 'pcm_s24le', \
        '-/filter_complex', volume_filter_f.name,
        output_audio_fname])

    # Remove the temporary file with our filter definitions.
    os.remove(volume_filter_f.name)
    return output_audio_fname

# Create an anonymized version of the given video file, storing the result in
# the given output directory with the given suffix sandwiched between the
# file's basename and its file extension.
# 
# At each of the given segments, this method will (a) replace the audio with
# silence and (b) replace the video with a box blur covering the entire frame,
# such that the participants can neither be seen nor heard.
def anonymize_video(input_video_fname, segments_to_anonymize, output_dir, \
                    output_suffix):
    # Construct a string of filter commands for ffmpeg(1) that sets the volume
    # to zero (= silences the audio) and applies a full-frame box blur at the
    # intervals specified in the tuples given in 'segments_to_anonymize'.
    # This should look something like:
    #
    #   [0:v]format=yuv420p,
    #        boxblur=10:enable='between(t\,216.670\,219.193)',
    #        boxblur=10:enable='between(t\,351.430\,352.264)'; 
    #   [0:a]volume=enable='between(t\,216.670\,219.193)':volume=0, 
    #        volume=enable='between(t\,351.430\,352.264)':volume=0
    #
    # These definitions can get longer than the command line can handle, so we
    # write them to a named temporary file that ffmpeg(1) can then read using
    # its '-/filter_complex' option.

    # Start with the video-related filter options.
    video_filter = 'format=yuv420p, '
    video_filter += ', '.join(\
        ["boxblur=10:enable='between(t\,%.3f\,%.3f)'" % \
         (start_ms / 1000.0, end_ms / 1000.0) for \
         (start_ms, end_ms, _) in segments_to_anonymize])
    video_filter += '; '

    # ...then the audio-related filters.
    video_filter += ', '.join(\
        ["volume=enable='between(t\,%.3f\,%.3f)':volume=0" % \
         (start_ms / 1000.0, end_ms / 1000.0) for \
         (start_ms, end_ms, _) in segments_to_anonymize])

    # Write the filters to our temporary filter script.
    video_filter_f = tempfile.NamedTemporaryFile(mode = 'w', \
        delete = False, suffix = '.txt')
    video_filter_f.write(video_filter)
    video_filter_f.close()

    # Store the anonymized video in 'output_dir' under the name '${BASENAME}-
    # ${OUTPUT_SUFFIX}.mp4" by default.
    output_video_fname = os.path.join(output_dir, \
        os.path.splitext(os.path.basename(input_video_fname))[0] + \
        output_suffix + '.mp4')

    # Process the video using ffmpeg(1).
    subprocess.call(['ffmpeg', \
        '-y', \
        '-v', '0', \
        '-i', input_video_fname, \
        '-c:v', 'libx264', \
        '-preset', 'veryslow', \
        '-crf', '17', \
        '-/filter_complex', video_filter_f.name,
        '-movflags', '+faststart', \
        output_video_fname])

    # Remove the temporary file with our filter definitions.
    os.remove(video_filter_f.name)
    return output_video_fname

# Use ffmpeg(1) to return the duration of the given media file (in
# milliseconds), or -1 if the duration could not be retrieved.
def get_duration_ms(media_fname):
    clip_duration_ms = -1

    try:
        clip_info = str(subprocess.check_output(['ffmpeg', '-i', media_fname, \
            '-af', 'astats', '-f', 'null', '-'], stderr = subprocess.STDOUT))

        clip_samples_m = re.search("Number of samples: (\d+)", clip_info)
        clip_samples = int(clip_samples_m.group(1))

        clip_sample_rate_m = re.search("Audio: .*?, (\d+) Hz,", clip_info)
        clip_sample_rate = int(clip_sample_rate_m.group(1))

        clip_duration_ms = \
            int(round(clip_samples / (clip_sample_rate / 1000.0)))
    except subprocess.CalledProcessError as e:
        print("ERROR: couldn't retrieve audio info for '%s' (%d, %s)" % \
            (media_fname, e.returncode, e.output))
    except:
        print("ERROR: couldn't retrieve audio info for '%s'" % media_fname)

    return clip_duration_ms

# Convert milliseconds to either an ffmpeg(1)-style "HH:MM:SS.MSS" timestamp
# (default) or "HHhMMmSSsMSS" (which is often easier to include as part of
# file names).
def ms_to_timestamp(time_ms, ffmpeg = True):
    secs, millis = divmod(time_ms, 1000)
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    if ffmpeg:
        return '%02d:%02d:%02d.%03d' % (hours, mins, secs, millis)

    return '%02dh%02dm%02ds%03d' % (hours, mins, secs, millis)

def create_clip(input_fname, start_ms, end_ms, output_fname):
    # Create a clip using ffmpeg(1).  This currently transcodes the video at
    # a lower quality, which is (much) slower than simply copying the existing
    # streams, but this avoids the problem of blank video frames for clips
    # starting before an I-frame.
    subprocess.call(['ffmpeg', \
        '-y', \
        '-v', '0', \
        '-fflags', '+genpts', \
        '-i', input_fname, \
        '-ss', ms_to_timestamp(start_ms), \
        '-t', str((end_ms - start_ms) / 1000.0), \
        '-acodec', 'copy', \
        '-vcodec', 'libx264', \
        '-preset', 'veryfast', \
        '-avoid_negative_ts', '1', \
        output_fname])

# Create individual clips of each of the given segments from each of the
# anonymized output files in the given output_dir.
def review_anonymization(segments_to_anonymize, output_dir):
    for media in [os.path.join(output_dir, f) for f in os.listdir(output_dir) \
                  if re.search(r'.*\.(mp4|wav)$', f)]:
        media_dur_ms = get_duration_ms(media)
        media_fname, media_ext = os.path.splitext(media)

        for (start_ms, end_ms, _) in segments_to_anonymize:
            clip_fname = "%s-%s_%s%s" % \
                (media_fname, ms_to_timestamp(start_ms, False), \
                 ms_to_timestamp(end_ms, False), media_ext)

            # Include a bit of context before and after each clip.
            start_ms = max(start_ms - REVIEW_CONTEXT, 0)
            end_ms = min(end_ms + REVIEW_CONTEXT, media_dur_ms)

            # Create a clip using ffmpeg(1).
            create_clip(media, start_ms, end_ms, clip_fname)

# Pre-compile a regular expression to determine whether or not a given string
# contains one of the three mark-up tags ("[anon (...)]", "[name]", or 
# "[sensitive]") that are used to indicate text that needs to be anonymized.
anonymous_text_re = re.compile('\[\/?(anon|name|sensitive)')


#
# 1. Read in all of the options from the command line.
#
parser = argparse.ArgumentParser(description = \
    'Anonymize the text and media associated with the given ELAN transcripts.')
parser.add_argument('elan_transcripts', nargs = '+', help = \
    'ELAN transcripts to anonymize')
parser.add_argument('-o', '--output_dir', default = 'anonymized', help = \
    'Directory in which to store output anonymized transcripts and media')
parser.add_argument('-s', '--output_suffix', default = '-ANONYMIZED', help = \
    'Suffix to add to output file basenames')
parser.add_argument('-r', '--review_anonymization', action = 'store_true',
    help = 'create stand-alone media clips in the output directory for '\
           'each anonymized segment')
parser.add_argument('-a', '--audio_suffix', default = '\.wav$', help = \
    'Regular expression added after the basename of the given ELAN '\
    'transcripts to identify associated audio files to anonymize')
parser.add_argument('-v', '--video_suffix', default = '\.mp4$', help = \
    'Regular expression added after the basename of the given ELAN '\
    'transcripts to identify associated audio files to anonymize')
parser.add_argument('-na', '--no_audio', action = 'store_true',
    help = 'do not produce anonymized copies of transcripts\' audio')
parser.add_argument('-nv', '--no_video', action = 'store_true',
    help = 'do not produce anonymized copies of transcripts\' video')
parser.add_argument('-nt', '--no_text', action = 'store_true',
    help = 'do not produce anonymized copies of the given ELAN transcripts')
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)


#
# 2. Anonymize each of the given transcripts and all associated media.
#

for input_transcript_fname in args.elan_transcripts:
    # Grab the absolute path to the directory in which this transcript is
    # stored (for locating all associated media), as well as the name of this
    # transcript with its file extension stripped off (again for easier
    # matching against related media files).
    transcript_dir = os.path.dirname(os.path.abspath(input_transcript_fname))
    transcript_base = \
        os.path.splitext(os.path.basename(input_transcript_fname))[0]

    transcript = pympi.Elan.Eaf(input_transcript_fname)
    print("Anonymizing transcript '%s'" % input_transcript_fname)

    for tier in transcript.tiers:
        # 'get_annotation_data_for_tier' can return either reference or
        # aligned annotations, which require different methods to change
        # the values of.  To help abstract away from this unpleasant detail,
        # we figure out (based on the length of the tuples that this method
        # returns) which kind of tier we're dealing with here, then choose
        # the appropriate method to call when changing the value of annotations
        # on this tier.
        annotations = transcript.get_annotation_data_for_tier(tier)
        set_annotation = set_aligned_annotation
        if annotations and len(annotations[0]) > 3:
            set_annotation = set_ref_annotation

        for (start_ms, end_ms, value, *refvalue) in annotations:
            # Change the text of this annotation iff it requires anonymization.
            if anonymous_text_re.search(value):
#                print(tier, start_ms, end_ms, value, anonymize_text(value))
                set_annotation(transcript, tier, start_ms, end_ms, \
                    anonymize_text(value))

    # List of tuples containing (start_ms, end_ms, text), indicating all of
    # the timestamps in the media that need to be anonymized.
    segments_to_anonymize = \
        transcript.get_annotation_data_for_tier('Postprocess')

    # Run through the annotations on the "Postprocess" tier and anonymize
    # (e.g., blur for video, bleep or silence for audio) those sections of
    # all of the associated media.
    original_to_anonymized_media = {}

    # Process all audio files in the same folder as the input transcript
    # and that share its basename (and are in WAV format).
    if not args.no_audio:
        for audio in [os.path.join(transcript_dir, f) for f in \
                os.listdir(transcript_dir) if \
                re.search(transcript_base + args.audio_suffix, f)]:
            print("Anonymizing audio file '%s'" % audio)
            anonymized_audio_fname = anonymize_audio(audio, \
                segments_to_anonymize, args.output_dir, args.output_suffix)
            original_to_anonymized_media[os.path.basename(audio)] = \
                anonymized_audio_fname

    # Process all video files in the same folder as the input transcript
    # and that share its basename (and are in MP4 format).
    if not args.no_video:
        for video in [os.path.join(transcript_dir, f) for f in \
                os.listdir(transcript_dir) if \
                re.search(transcript_base + args.video_suffix, f)]:
            print("Anonymizing video file '%s'" % video)
            anonymized_video_fname = anonymize_video(video, \
                segments_to_anonymize, args.output_dir, args.output_suffix)
            original_to_anonymized_media[os.path.basename(video)] = \
                anonymized_video_fname

    # If requested, create stand-alone clips of the output media for each of
    # the segments that needed to be anonymized with a bit of surrounding
    # context (e.g., 1-2 seconds), so that they can be reviewed more easily.
    if args.review_anonymization:
        print("Reviewing anonymization...")
        review_anonymization(segments_to_anonymize, args.output_dir)

    # Now save the anonymized transcript.
    if not args.no_text:
        # Update the names of the media files in the transcript to refer to
        # the anonymized versions.
        for media_descriptor in transcript.media_descriptors:
            media_fname = os.path.basename(media_descriptor['MEDIA_URL'])
            if media_fname in original_to_anonymized_media:
                anonymized_fname = original_to_anonymized_media[media_fname]
                media_descriptor['MEDIA_URL'] = \
                    media_descriptor['MEDIA_URL'].removesuffix(media_fname) + \
                    anonymized_fname
                media_descriptor['RELATIVE_MEDIA_URL'] = \
                    media_descriptor['RELATIVE_MEDIA_URL'].removesuffix(\
                    media_fname) + anonymized_fname

        output_transcript_fname, output_transcript_ext = \
            os.path.splitext(os.path.basename(input_transcript_fname))
        output_transcript_fname = "%s%s%s" % \
            (output_transcript_fname, args.output_suffix, output_transcript_ext)
        pympi.Elan.to_eaf(\
            os.path.join(args.output_dir, output_transcript_fname), transcript)

    print()
