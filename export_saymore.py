#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Create an ELAN transcript to annotate the three-channel WAV file that SayMore
# generates for orally annotated media, based on the contents of the ELAN file
# in the SayMore session.
# 
# Optionally (though preferably), this script can also generate that combined,
# three-channel WAV audio itself, taking care of anonymization in both the
# output ELAN transcript *and* the audio in the process.
#
#
# Usage:
#
#   ./export_saymore.py -a -t "Gerald Meguinis" srs-DCT-20181201-02-01/srs-DCT-20181201-02-01_Source.wav.annotations-FIXED.eaf
#
#   ./export_saymore.py -a -A -t "Gerald Meguinis" srs-DCT-20181201-02-01/srs-DCT-20181201-02-01_Source.wav.annotations-FIXED-WITH_ANON.eaf

import argparse
import codecs
import os.path
import re
import subprocess
import sys
import tempfile
import urllib.request

import mako
import mako.exceptions
import mako.template
import pydub
import pympi

# Convert millisecond start and end types and a type ('Translation'|'Careful')
# into a SayMore-style oral annotation clip file name.
def to_oa(start_ms, end_ms, oa_type):
    start_s = ('%.3f' %  (start_ms / 1000.0)).rstrip('0').rstrip('.')
    end_s = ('%.3f' %  (end_ms / 1000.0)).rstrip('0').rstrip('.')
    return '%s_to_%s_%s.wav' % (start_s, end_s, oa_type)

# Return the full, local path to the media specified in the given media
# descriptor in this ELAN transcript, or None if this file can't be located.
# (If neither the absolute or relative media URLs given in the transcript
# get us back to the media, we also check as a last resort to see if the
# media might be in the same directory as its transcript)
def find_local_media(transcript_fname, media_descriptor):
    media_file_location = None
    media_url = media_descriptor.get('MEDIA_URL', '')
    transcript_dir = os.path.dirname(os.path.abspath(transcript_fname))

    # SayMore sometimes stores plain file names (e.g., "file.wav"), rather
    # than URL-encoded paths (e.g., "file:///path/to/file.wav") or other
    # local paths in MEDIA_URL.  Check for this first.
    if os.path.isfile(media_url):
        media_file_location = os.path.abspath(media_url)
    elif os.path.isfile(os.path.join(transcript_dir, media_url)):
        media_file_location = \
            os.path.abspath(os.path.join(transcript_dir, media_url))
    else:
        try:
            # Look for the file under its MEDIA_URL first.
            (media_file_location, _) = urllib.request.urlretrieve(\
                media_descriptor.get('MEDIA_URL', ''))
        except Exception:
            # If that doesn't work, look under its RELATIVE_MEDIA_URL (if one
            # exists).
            media_file_location = os.path.join(transcript_dir, \
                media_descriptor.get('RELATIVE_MEDIA_URL', ''))
            if not os.path.isfile(media_file_location):
                # If neither the MEDIA_URL or the RELATIVE_MEDIA_URL get us to
                # the media, check in the same directory as the transcript
                # itself (on the off chance that the media and the transcript
                # have been copied there without the file locations in the
                # transcript header getting updated)
                media_fname = media_file_location.split('/')[-1]
                media_file_location = os.path.join(transcript_dir, media_fname)
                if not os.path.isfile(media_file_location):
                    media_file_location = None

    return media_file_location

# Replace any text that is marked up as needing to be anonymized with
# appropriate placeholders.
def _anonymize_text(s):
    # '[anon type="name"]...[/anon]' or '[name]...[/name]' --> '(name)'
    s = re.sub('\[anon type="name"\](.*?)\[\/anon\]', '(NAME)', s)
    s = re.sub('\[name\](.*?)\[\/name\]', '(NAME)', s)

    # '[anon type="language|topic|sensitive"]...[/anon]' or 
    # '[sensitive]...[/sensitive]' --> '(sensitive)'
    s = re.sub('\[anon type=".*?"\](.*?)\[\/anon\]', '(SENSITIVE)', s)
    s = re.sub('\[sensitive\](.*?)\[\/sensitive\]', '(SENSITIVE)', s)

    return s

def anonymize(s):
    anon_s = _anonymize_text(s)
    return anon_s, anon_s != s


# Read in all of the options from the command line.
parser = argparse.ArgumentParser(description = \
    'Export an ELAN transcript for SayMore\'s generated oral annotation audio')
parser.add_argument('elan_transcript', help = 'Source SayMore ELAN transcript')
parser.add_argument('-oa', '--original-annotator', \
    default = 'Christopher Cox', \
    help = 'Name(s) of the contributor(s) who provided textual annotations '\
           'for the source annotations in this oral annotation session')
parser.add_argument('-r', '--repeater', default = 'Bruce R. Starlight', \
    help = 'Name of the contributor who provided careful repetitions in '\
           'this oral annotation session')
parser.add_argument('-ra', '--repetition-annotator', \
    default = 'Christopher Cox', \
    help = 'Name(s) of the contributor(s) who provided textual annotations '\
           'for the careful repetitions in this oral annotation session')
parser.add_argument('-t', '--translator', default = 'Bruce R. Starlight', \
    help = 'Name of the contributor who provided free translations in this '\
           'oral annotation session')
parser.add_argument('-ta', '--translation-annotator', \
    default = 'Christopher Cox, Alaa Sarji', \
    help = 'Name(s) of the contributor(s) who provided textual annotations '\
           'for the careful repetitions in this oral annotation session')
parser.add_argument('-a', '--generate-audio', action = 'store_true', \
    help = 'Generate combined audio tracks for this oral annotation session')
parser.add_argument('-A', '--anonymize', action = 'store_true', \
    help = 'Anonymize the text in the exported ELAN transcript (and, if '\
    '--generate_audio is specified, in the combined audio tracks, as well)')
parser.add_argument('-d', '--output-dir',
    help = 'Directory in which to store all output files. If not specified, '\
    'this defaults to the directory in which the given ELAN transcript is '\
    'located')
parser.add_argument('-o', '--output-prefix',
    help = 'Use this string as the first component in the name of all output '\
    'files produced by this script. If not specified, this defaults to the '\
    'name of the SayMore session, followed by ".oralannotations" and a '\
    'file extension appropriate for the type of output being produced.')
args = parser.parse_args()

# Load the SayMore ELAN transcript.
if not os.path.isfile(args.elan_transcript):
    print("ERROR: unable to open source SayMore ELAN transcript: '%s'" % \
        args.elan_transcript)
    sys.exit(1)

src_transcript = pympi.Elan.Eaf(args.elan_transcript)
src_text_tier = \
    src_transcript.get_tier_ids_for_linguistic_type('Transcription')[0]
src_translation_tier = \
    src_transcript.get_tier_ids_for_linguistic_type('Translation')[0]
src_source_tier = \
    src_transcript.get_tier_ids_for_linguistic_type('SayMoreify-Metadata')[0]

# Get the name of the source audio (so that we can open and iterate through
# the contents of the corresponding oral annotations folder, which has the
# name ${src_audio}_Annotations).
src_audio_fname = None
for media_descriptor in src_transcript.media_descriptors:
    media_file = find_local_media(args.elan_transcript, media_descriptor)
    print(media_file)
    if media_file and media_file.lower().endswith('.wav'):
        src_audio_fname = os.path.join(os.path.dirname(args.elan_transcript), \
            os.path.basename(media_file))
        break

assert src_audio_fname, f"ERROR: can't locate SayMore ELAN transcript audio '{src_audio_fname}'"

if args.output_prefix:
    dst_audio_fname = os.path.join(os.path.dirname(src_audio_fname), 
        f"{args.output_prefix}.wav")
else:
    dst_audio_fname = src_audio_fname + '.oralAnnotations.wav'

if args.output_dir:
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    dst_audio_fname = os.path.join(os.path.abspath(args.output_dir),
        os.path.basename(dst_audio_fname))

oa_dir = src_audio_fname + '_Annotations'
assert os.path.isdir(oa_dir), f"ERROR: no oral annotations directory '{oa_dir}'"

# Load the source audio (so that we can generate clips from it later on).
src_audio = pydub.AudioSegment.from_wav(src_audio_fname)

# Use the contents of the SayMore ELAN transcript to create a list of 
# annotations containing the audio (original and oral annotations) and
# text available for all (non-ignored) segments,  This list has the format:
#
# [
#   {
#       'orig':             '...', # Transcription of original audio
#       'rep':              '...', # Transcription of careful repetition
#       'trans':            '...', # Transcription of free translation
#       'source':           '...', # Source information
#       'audio_orig':       <audio clip of source segment>,
#       'audio_rep':        <audio clip of careful repetition, if it exists>,
#       'audio_trans':      <audio clip of free translation, if it exists>
#    },
#    ...
# ]
annotations = []
offset = 0
annotation_id = 1
timeslot_id = 1
for (src_start_ms, src_end_ms, text) in \
    sorted(src_transcript.get_annotation_data_for_tier(src_text_tier)):

    # Skip any annotations that are marked in SayMore as needing to be
    # ignored -- those aren't exported by SayMore in its generated OA audio,
    # so we don't need to worry about representing them in the matching ELAN
    # transcript that this script produces.
    if text == '%ignore%':
        continue

    annotation = {
        'orig-anonymized':  False,
        'rep-anonymized':   False,
        'trans-anonymized': False,
    }

    # In the transcription conventions for our ELDP project, separate
    # transcriptions of the source audio *and* its supposed careful repetition
    # are stored in annotations on the 'Transcription' tier, and are
    # separated by ' || ' (original first, repetition second).
    #
    # TODO: Not every project uses their 'Transcription' annotations like this.
    # Add an argument to turn this parsing on/off.
    if ' || ' in text:
        annotation['orig'], annotation['rep'] = text.split(' || ')
    else:
        annotation['orig'] = text
        annotation['rep'] = ''
    
    # Get the transcription of the free translation and source information for
    # this annotation.
    _, _, annotation['trans'], _ = src_transcript.get_ref_annotation_at_time(\
        src_translation_tier, (src_start_ms + src_end_ms) / 2)[0]
    _, _, annotation['source'], _ = src_transcript.get_ref_annotation_at_time(\
        src_source_tier, (src_start_ms + src_end_ms) / 2)[0]

    # If we've been asked to anonymize our output, try to do that to the text
    # that we have now.
    if args.anonymize:
        a = annotation
        a['orig'], a['orig-anonymized'] = anonymize(a['orig'])
        a['rep'], a['rep-anonymized'] = anonymize(a['rep'])
        a['trans'], a['trans-anonymized'] = anonymize(a['trans'])

    # Now that we've extracted all of the text, we need to deal with audio
    # clips for the original audio, the careful repetition (if one exists),
    # and the free translation (if one exists), too.
    annotation['audio_orig'] = src_audio[src_start_ms:src_end_ms]

    # Nasty hack: keep track of timeslots and unique annotation IDs now, so
    # that we don't have to do it in the Mako template below when generating
    # an ELAN transcript.
    annotation['audio_orig_ts1'] = offset
    annotation['audio_orig_ts1id'] = 'ts%d'  % timeslot_id
    annotation['audio_orig_ts2'] = offset + (src_end_ms - src_start_ms)
    annotation['audio_orig_ts2id'] = 'ts%d' % (timeslot_id + 1)
    annotation['audio_orig_aid'] = 'a%d' % annotation_id
    offset += (src_end_ms - src_start_ms)
    annotation_id += 1
    timeslot_id += 2

    annotation['audio_rep'] = None
    audio_rep_fname = os.path.join(oa_dir, \
        to_oa(src_start_ms, src_end_ms, 'Careful'))
    if os.path.isfile(audio_rep_fname):
        annotation['audio_rep'] = pydub.AudioSegment.from_wav(audio_rep_fname)

        annotation['audio_rep_ts1'] = offset
        annotation['audio_rep_ts1id'] = 'ts%d' % timeslot_id
        annotation['audio_rep_ts2'] = offset + len(annotation['audio_rep'])
        annotation['audio_rep_ts2id'] = 'ts%d' % (timeslot_id + 1)
        annotation['audio_rep_aid'] = 'a%d' % annotation_id
        offset += len(annotation['audio_rep'])
        annotation_id += 1
        timeslot_id += 2

    annotation['audio_trans'] = None
    audio_trans_fname = os.path.join(oa_dir, \
        to_oa(src_start_ms, src_end_ms, 'Translation'))
    if os.path.isfile(audio_trans_fname):
        annotation['audio_trans'] = \
            pydub.AudioSegment.from_wav(audio_trans_fname)

        annotation['audio_trans_ts1'] = offset
        annotation['audio_trans_ts1id'] = 'ts%d' % timeslot_id
        annotation['audio_trans_ts2'] = offset + len(annotation['audio_trans'])
        annotation['audio_trans_ts2id'] = 'ts%d' % (timeslot_id + 1)
        annotation['audio_trans_aid'] = 'a%d' % annotation_id
        offset += len(annotation['audio_trans'])
        annotation_id += 1
        timeslot_id += 2

    # Now that we have all of the audio loaded for this annotation, if we've
    # been asked to do any anonymization, replace the audio for any annotations
    # that contain anonymized text with a period of silence of the same
    # duration.
    #
    # (It's a shame to have to leave out so much audio when only a single word
    # may need to be omitted, but short of implementing some form of forced
    # alignment (and trusting it to do a reliable enough job to be suitable
    # for a relatively high-stakes task like anonymization), we really don't
    # have any decent alternatives right now, since we don't have any way of
    # referring permanently to stretches of audio in anything but the original
    # recordings themselves.  Think about this more and see if this could be
    # addressed in future work.)
    if args.anonymize:
        if annotation['orig-anonymized']:
            annotation['audio_orig'] = pydub.AudioSegment.silent(\
                duration = len(annotation['audio_orig']))

#        if annotation['rep-anonymized']:
        if annotation['rep-anonymized'] and annotation['audio_rep']:
            annotation['audio_rep'] = pydub.AudioSegment.silent(\
                duration = len(annotation['audio_rep']))

#        if annotation['trans-anonymized']:
        if annotation['trans-anonymized'] and annotation['audio_trans']:
            annotation['audio_trans'] = pydub.AudioSegment.silent(\
                duration = len(annotation['audio_trans']))

    annotations.append(annotation)

# Finally, produce an ELAN transcript for the generated WAV file that
# SayMore exports from sessions with oral annotations. This contains four
# top-level tiers:
#
#   1. 'Original', for the transcribed text of the original audio;
#   2. 'Repetition', for the transcribed text of careful repetitions;
#   3. 'Translation', for the transcribed text of free translations;
#   4. 'Postprocess', for later anonymization work.
#
# There are also child tiers for the first three of these, assigning a number
# [1-n] to each one to associate sets of related annotations together (e.g.,
# the original audio, careful repetition, and free translation annotations for
# the first non-ignored segment in the original SayMore transcript would all
# receive the number '0' in an annotatino on their *-ID tier).  Basically:
#
#       Original ['text', lang: srs, Participant: ___; Annotator: CDC]
#           Original-ID ['oral-annotation-id']
#           Original-Source ['oral-annotation-source']
#       Repetition ['text', lang: srs, Participant: ___; Annotator: CDC]
#           Repetition-ID ['oral-annotation-id']
#       Translation ['text', lang: eng, Participant: ___; Annotator: CDC, AS/CH]
#           Translation-ID ['oral-annotation-id']
#       Postprocess ['event']

EAF_TEMPLATE = \
"""<?xml version="1.0" encoding="UTF-8"?>
<%
import cgi, os, random, time

datestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
datestamp = datestamp[:-2] + ':' + datestamp[-2:]
%>\\
<ANNOTATION_DOCUMENT AUTHOR="" DATE="${datestamp}" FORMAT="3.0" VERSION="3.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv3.0.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="file://${os.path.abspath(output_audio_fname)}" MIME_TYPE="audio/x-wav" RELATIVE_MEDIA_URL="./${os.path.basename(output_audio_fname)}"/>
<%
def get_random_hex(n = 1):
    s = ''
    for i in range (0, n):
        s += random.choice('0123456789abcdef')
    return s

def get_elan_urn():
    return 'urn:nl-mpi-tools-elan-eaf:' + \\
        get_random_hex(8) + '-' + \\
        get_random_hex(4) + '-' + \\
        get_random_hex(4) + '-' + \\
        get_random_hex(4) + '-' + \\
        get_random_hex(12)
%>\\
        <PROPERTY NAME="URN">${get_elan_urn()}</PROPERTY>
        <PROPERTY NAME="lastUsedAnnotationId"></PROPERTY>
    </HEADER>
    <TIME_ORDER>
<% last_ann = last_annotation_id %>\\
%for a in annotations:
        <TIME_SLOT TIME_SLOT_ID="${a['audio_orig_ts1id']}" TIME_VALUE="${a['audio_orig_ts1']}"/>
        <TIME_SLOT TIME_SLOT_ID="${a['audio_orig_ts2id']}" TIME_VALUE="${a['audio_orig_ts2']}"/>
%if a['audio_rep']:
        <TIME_SLOT TIME_SLOT_ID="${a['audio_rep_ts1id']}" TIME_VALUE="${a['audio_rep_ts1']}"/>
        <TIME_SLOT TIME_SLOT_ID="${a['audio_rep_ts2id']}" TIME_VALUE="${a['audio_rep_ts2']}"/>
%endif
%if a['audio_trans']:
        <TIME_SLOT TIME_SLOT_ID="${a['audio_trans_ts1id']}" TIME_VALUE="${a['audio_trans_ts1']}"/>
        <TIME_SLOT TIME_SLOT_ID="${a['audio_trans_ts2id']}" TIME_VALUE="${a['audio_trans_ts2']}"/>
%endif
%endfor
    </TIME_ORDER>
    <TIER LANG_REF="srs" LINGUISTIC_TYPE_REF="oral-annotation-text" ANNOTATOR="${original_annotator}" TIER_ID="Original">
%for a in annotations:
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="${a['audio_orig_aid']}" TIME_SLOT_REF1="${a['audio_orig_ts1id']}" TIME_SLOT_REF2="${a['audio_orig_ts2id']}">
                <ANNOTATION_VALUE>${a['orig']}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
%endfor
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="oral-annotation-id" PARENT_REF="Original" TIER_ID="Original-ID">
%for (i, a) in enumerate(annotations):
        <ANNOTATION>
            <REF_ANNOTATION ANNOTATION_ID="a${last_ann}" ANNOTATION_REF="${a['audio_orig_aid']}">
                <ANNOTATION_VALUE>${i}</ANNOTATION_VALUE>
            </REF_ANNOTATION>
        </ANNOTATION>
<% last_ann += 1 %>\\
%endfor
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="oral-annotation-source" PARENT_REF="Original" TIER_ID="Original-Source">
%for a in annotations:
        <ANNOTATION>
            <REF_ANNOTATION ANNOTATION_ID="a${last_ann}" ANNOTATION_REF="${a['audio_orig_aid']}">
                <ANNOTATION_VALUE>${a['source']}</ANNOTATION_VALUE>
            </REF_ANNOTATION>
        </ANNOTATION>
<% last_ann += 1 %>\\
%endfor
    </TIER>
    <TIER LANG_REF="srs" LINGUISTIC_TYPE_REF="oral-annotation-text" PARTICIPANT="${repeater}" ANNOTATOR="${repetition_annotator}" TIER_ID="Repetition">
%for a in annotations:
%if a['audio_rep']:
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="${a['audio_rep_aid']}" TIME_SLOT_REF1="${a['audio_rep_ts1id']}" TIME_SLOT_REF2="${a['audio_rep_ts2id']}">
                <ANNOTATION_VALUE>${a['rep']}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
%endif
%endfor
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="oral-annotation-id" PARENT_REF="Repetition" TIER_ID="Repetition-ID">
%for (i, a) in enumerate(annotations):
%if a['audio_rep']:
        <ANNOTATION>
            <REF_ANNOTATION ANNOTATION_ID="a${last_ann}" ANNOTATION_REF="${a['audio_rep_aid']}">
                <ANNOTATION_VALUE>${i}</ANNOTATION_VALUE>
            </REF_ANNOTATION>
        </ANNOTATION>
<% last_ann += 1 %>\\
%endif
%endfor
    </TIER>
    <TIER LANG_REF="eng" LINGUISTIC_TYPE_REF="oral-annotation-text" PARTICIPANT="${translator}" ANNOTATOR="${translation_annotator}" TIER_ID="Translation">
%for a in annotations:
%if a['audio_trans']:
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="${a['audio_trans_aid']}" TIME_SLOT_REF1="${a['audio_trans_ts1id']}" TIME_SLOT_REF2="${a['audio_trans_ts2id']}">
                <ANNOTATION_VALUE>${a['trans']}</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
%endif
%endfor
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="oral-annotation-id" PARENT_REF="Translation" TIER_ID="Translation-ID">
%for (i, a) in enumerate(annotations):
%if a['audio_trans']:
        <ANNOTATION>
            <REF_ANNOTATION ANNOTATION_ID="a${last_ann}" ANNOTATION_REF="${a['audio_trans_aid']}">
                <ANNOTATION_VALUE>${i}</ANNOTATION_VALUE>
            </REF_ANNOTATION>
        </ANNOTATION>
<% last_ann += 1 %>\\
%endif
%endfor
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="event" TIER_ID="Postprocess" />
    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="oral-annotation-text" TIME_ALIGNABLE="true"/>
    <LINGUISTIC_TYPE CONSTRAINTS="Symbolic_Association" GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="oral-annotation-id" TIME_ALIGNABLE="false"/>
    <LINGUISTIC_TYPE CONSTRAINTS="Symbolic_Association" GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="oral-annotation-source" TIME_ALIGNABLE="false"/>
    <LINGUISTIC_TYPE GRAPHIC_REFERENCES="false" LINGUISTIC_TYPE_ID="event" TIME_ALIGNABLE="true"/>
    <LANGUAGE LANG_DEF="http://cdb.iso.org/lg/CDB-00138502-001" LANG_ID="eng" LANG_LABEL="English (eng)"/>
    <LANGUAGE LANG_DEF="http://cdb.iso.org/lg/CDB-00137345-001" LANG_ID="srs" LANG_LABEL="Sarsi (srs)"/>
    <CONSTRAINT DESCRIPTION="Time subdivision of parent annotation's time interval, no time gaps allowed within this interval" STEREOTYPE="Time_Subdivision"/>
    <CONSTRAINT DESCRIPTION="Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered" STEREOTYPE="Symbolic_Subdivision"/>
    <CONSTRAINT DESCRIPTION="1-1 association with a parent annotation" STEREOTYPE="Symbolic_Association"/>
    <CONSTRAINT DESCRIPTION="Time alignable annotations within the parent annotation's time interval, gaps are allowed" STEREOTYPE="Included_In"/>
</ANNOTATION_DOCUMENT>"""

variables = { \
    'repeater' : args.repeater, \
    'translator' : args.translator, \
    'original_annotator' : args.original_annotator, \
    'repetition_annotator' : args.repetition_annotator, \
    'translation_annotator' : args.translation_annotator, \

    'output_audio_fname' : dst_audio_fname, \
    'last_annotation_id' : annotation_id + 1, \
    'annotations' : annotations
}

mako_template = mako.template.Template(EAF_TEMPLATE, \
    input_encoding = 'utf-8', output_encoding = 'utf-8')

if args.output_prefix:
    output_file_name = os.path.join(os.path.dirname(dst_audio_fname), 
        f'{args.output_prefix}.eaf')
else:
    output_file_name = f'{dst_audio_fname}.annotations.eaf'

output_file = codecs.open(output_file_name, 'w', 'utf-8')

try:
    output_file.write(mako_template.render_unicode(**variables))
except:
    print(mako.exceptions.text_error_template().render())
    sys.exit(-1)

output_file.close()

if args.generate_audio:
    out_orig = pydub.AudioSegment.empty()
    out_rep = pydub.AudioSegment.empty()
    out_trans = pydub.AudioSegment.empty()
    
    for a in annotations:
        clip_orig = a['audio_orig']
        silent_orig = pydub.AudioSegment.silent(duration = len(clip_orig))

        clip_rep = a['audio_rep']
        if clip_rep:
            silent_rep = pydub.AudioSegment.silent(duration = len(clip_rep))
        else:
            clip_rep = pydub.AudioSegment.empty()
            silent_rep = clip_rep

        clip_trans = a['audio_trans']
        if clip_trans:
            silent_trans = pydub.AudioSegment.silent(duration = len(clip_trans))
        else:
            clip_trans = pydub.AudioSegment.empty()
            silent_trans = clip_trans

        out_orig += clip_orig + silent_rep + silent_trans
        out_rep += silent_orig + clip_rep + silent_trans
        out_trans += silent_orig + silent_rep + clip_trans

    with tempfile.NamedTemporaryFile(suffix = '.wav') as joined_orig, \
         tempfile.NamedTemporaryFile(suffix = '.wav') as joined_rep, \
         tempfile.NamedTemporaryFile(suffix = '.wav') as joined_trans:
        out_orig.export(joined_orig.name, format = 'wav')
        out_rep.export(joined_rep.name, format = 'wav')
        out_trans.export(joined_trans.name, format = 'wav')

        # Create a three-channel WAV file in the exact same format as SayMore 
        # produces (i.e., containing with three mono tracks representing the
        # original audio, the careful repetition, and the free translation, in
        # that order).
        #
        # (In fact, ffmpeg(1) reports that SayMore saves its three-channel
        # WAV files with a 2.1 channel layout, rather than the 3.0 layout 
        # specified below.  Exporting with that same layout using the following
        # filter definition
        #
        #     '[0:a][1:a][2:a]join=inputs=3:channel_layout=2.1'
        #
        # produces a three-track WAV file that Audacity plays without any
        # trouble, but that ELAN and macOS Finder's built-in preview struggle
        # to play properly, leaving out the third channel entirely.  We'll
        # stick to exporting 3.0 for now, though this may be worth looking
        # into further down the road.)
        subprocess.run([\
            'ffmpeg', \
                '-y', \
                '-v', '0', \
                '-i', joined_orig.name, \
                '-i', joined_rep.name, \
                '-i', joined_trans.name, \
                '-filter_complex', \
                '[0:a][1:a][2:a]join=inputs=3:channel_layout=3.0', \
                dst_audio_fname
        ])
