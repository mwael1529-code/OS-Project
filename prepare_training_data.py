from pathlib import Path
import shutil, json, re, textwrap, zipfile, sys, subprocess
import pandas as pd

# Portable: run from the project folder containing training_examples_v5_original.csv.
OUT = Path(__file__).resolve().parent
SRC = OUT
if not (OUT / 'training_examples_v5_original.csv').is_file():
    raise FileNotFoundError('Keep training_examples_v5_original.csv beside this script.')

# -------------------- Build expanded, balanced dataset --------------------
def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', s)
    s = re.sub(r'[إأآٱ]', 'ا', s)
    s = re.sub(r'[،؛؟]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

base_path = OUT / 'training_examples.csv'
base = pd.read_csv(SRC / 'training_examples_v5_original.csv')
base = base[['text','intent']].dropna().copy()
base['text'] = base['text'].astype(str).str.strip()
base = base[base['text'].ne('')]

# Existing bare play-song examples belong to the local-media feature, not YouTube search.
def should_move_to_media(text):
    t = norm(text)
    if re.search(r'\b(?:on youtube|youtube)\b|يوتيوب|اليوتيوب|music video by|videos? about|video about', t):
        return False
    # Topic-oriented video requests remain searches; title-style playback defaults local.
    if re.search(r'(?:فيديو|فيديوهات)\s+عن\b', t):
        return False
    return bool(re.match(r'^(?:play\s+(?:the\s+)?(?:song|track|movie|film|video|music)?\s*|شغل(?:لي|ني)?\s*(?:اغنية|أغنية|فيلم|فيديو|مقطع)?\s*|اسمع(?:لي)?\s*(?:اغنية|أغنية)?\s*)\S+', t))

for idx, row in base.iterrows():
    if str(row['intent']) == 'SEARCH_YOUTUBE' and should_move_to_media(row['text']):
        # Explicit media-title playback defaults to local library unless the user says YouTube.
        base.at[idx, 'intent'] = 'PLAY_MEDIA'

# Curated additions before templated expansion. Keep the wording useful and realistic.
SUPPLEMENT = {
'OPEN_APP': [
    'Could you open Firefox for me?', 'Please launch the browser', 'I need the file manager opened',
    'Start the terminal application', 'Open the settings window', 'Launch the system monitor',
    'Please run the calculator app', 'Bring up the text editor', 'Open the document viewer',
    'Start the music player', 'Open the software center', 'Launch the image viewer',
    'Can you start Visual Studio Code?', 'Would you run Firefox?', 'I want to open the terminal',
    'Open the application menu app', 'Please bring up Files', 'Run the calculator',
    'ممكن تفتحلي فايرفوكس', 'لو سمحت افتح التيرمنال', 'شغل مدير الملفات', 'افتح الإعدادات',
    'عايز افتح الآلة الحاسبة', 'ممكن تشغل في اس كود', 'افتح محرر النصوص', 'شغل مراقب النظام',
    'افتح مدير البرامج', 'عايزة افتح المتصفح', 'شغل برنامج الملفات', 'ممكن تفتح البرنامج',
    'عاوز افتح الفايرفوكس', 'افتح تطبيق الآلة الحاسبة', 'افتح برنامج عرض المستندات',
],
'CLOSE_APP': [
    'Could you close Firefox for me?', 'Please quit the browser', 'Exit the terminal app',
    'Close the settings window', 'Quit the system monitor', 'Please close the calculator',
    'Close the text editor application', 'Exit the document viewer', 'Stop the music player app',
    'Can you close Visual Studio Code?', 'Would you quit Firefox?', 'Close the active application',
    'Please exit the file manager', 'I want to close the terminal', 'Quit the software center',
    'اقفل فايرفوكس من فضلك', 'ممكن تقفل التيرمنال', 'اغلق مدير الملفات', 'اقفل الإعدادات',
    'اقفل الآلة الحاسبة', 'ممكن تقفل في اس كود', 'اقفل محرر النصوص', 'اغلق مراقب النظام',
    'عايز اقفل المتصفح', 'اقفل التطبيق الحالي', 'اقفل برنامج الملفات', 'ممكن تنهي البرنامج',
    'اقفل نافذة الإعدادات', 'اطلع من التيرمنال',
],
'OPEN_FOLDER': [
    'Open the folder called Projects', 'Please show the Downloads directory',
    'Go to my Documents folder', 'Open the directory named Reports',
    'Show me the Desktop directory', 'Open the Music directory', 'Go to Pictures',
    'Please open Videos', 'Open the folder for university assignments',
    'Show my project directory', 'Open the directory containing documents',
    'Take me to the Downloads folder', 'Open the work folder', 'Show the folder named Archive',
    'افتح مجلد التنزيلات لو سمحت', 'وديني على مجلد المستندات', 'افتح فولدر المشاريع',
    'وريني مجلد سطح المكتب', 'ادخل على مجلد الصور', 'افتح فولدر الموسيقى',
    'عايز افتح مجلد الفيديوهات', 'افتح فولدر الشغل', 'وريني مجلد الجامعة',
    'ممكن تفتح مجلد التقارير', 'افتح المجلد اللي اسمه الواجبات',
],
'SEARCH_FILE': [
    'Locate my project PDF', 'Locate my project file', 'Find the project document',
    'Search my computer for the report', 'Can you locate the assignment PDF?',
    'Look for my spreadsheet', 'Find the file named budget.xlsx', 'Search for the file called notes',
    'Where is my project file?', 'Where can I find the report PDF?', 'Look up my CV file',
    'Find the document in my folders', 'Search for the presentation file',
    'Find my downloaded invoice', 'Locate the latest report document',
    'Please find a file named statistics', 'Search Documents for my resume',
    'Find the spreadsheet called expenses', 'Locate the image file of the dashboard',
    'ممكن تدورلي على ملف المشروع PDF', 'فين ملف التقرير', 'ابحث عن ملف الواجب',
    'دورلي على ملف الإكسل', 'هات ملف السيرة الذاتية', 'فين ملف الباوربوينت',
    'اعثر على ملف اسمه الميزانية', 'دور على ملف المحاضرة', 'ابحث عن ملف الإحصاء',
    'ممكن تلاقي ملف المشروع', 'دور على ملف وورد', 'هات ملف الفاتورة',
    'فين صورة الداشبورد', 'ابحث عن ملف باسم التقارير', 'دورلي على ملف PDF للمشروع',
],
'GET_CPU': [
    'What is my current CPU utilization?', 'Tell me the processor load',
    'How busy is my processor right now?', 'Display the CPU percentage',
    'Get current processor statistics', 'How much processing power is being used?',
    'Show CPU activity', 'Give me the system processor usage', 'Check how heavily the CPU is working',
    'Measure the processor utilization', 'What percentage of CPU is active?',
    'اعرضلي نسبة استخدام المعالج', 'عايز اعرف ضغط البروسيسور', 'المعالج مستخدم كام بالمية؟',
    'هات حالة المعالج', 'وريني استهلاك السي بي يو', 'قيس استخدام المعالج',
    'شوف المعالج مشغول قد ايه', 'استهلاك البروسيسور عامل كام',
],
'GET_RAM': [
    'How much physical memory is in use?', 'Show available RAM in my computer',
    'Report current memory consumption', 'What is my memory utilization?',
    'Get current RAM statistics', 'How much free memory remains?',
    'Show the amount of used memory', 'Tell me the available system memory',
    'Check how much memory is left', 'Display RAM consumption',
    'ممكن تعرض الرام المتاحة', 'عايز اعرف استهلاك الذاكرة', 'فاضل كام ميجا في الرام؟',
    'وريني الرام المستخدمة', 'هات حالة الذاكرة', 'الذاكرة مستخدمة قد ايه؟',
    'اعرض نسبة الرام المستهلكة', 'قيس استخدام الذاكرة',
],
'GET_DISK': [
    'How much free disk capacity remains?', 'Report available drive space',
    'Show the storage capacity left', 'Tell me how much space is unused',
    'Check the available filesystem space', 'Get disk capacity statistics',
    'Show the free space on my drive', 'How full is my hard drive?',
    'Display remaining storage', 'Check free space on the system disk',
    'فاضل كام جيجا في الهارد؟', 'اعرض المساحة الفاضية على القرص',
    'عايز اعرف الهارد مليان قد ايه', 'هات المساحة المتاحة',
    'وريني مساحة التخزين الفاضية', 'اعرض بيانات مساحة القرص',
    'المساحة المتبقية على الجهاز كام', 'شوف القرص فاضل فيه كام',
],
'CHANGE_VOLUME': [
    'Raise the speaker volume', 'Decrease audio output', 'Set the sound level to 70 percent',
    'Make the speakers quieter', 'Increase audio output a little', 'Set volume to zero',
    'Unmute my speakers', 'Lower the media volume to 20 percent',
    'Turn the sound up', 'Reduce the volume by ten percent',
    'Raise the volume to 80', 'Make the audio louder please',
    'وطّي صوت السماعات', 'علي الصوت شوية', 'خلي مستوى الصوت ٧٠ في المية',
    'اكتم صوت الجهاز', 'رجع الصوت بعد الكتم', 'قلل الصوت لعشرين في المية',
    'ارفع مستوى السماعة', 'خلي الصوت أوطى', 'زود مستوى الصوت',
    'Set the system volume to 45 percent', 'Turn the speaker level down',
    'Set audio output to maximum', 'Set audio output to minimum',
    'Please silence the audio', 'Please restore the sound',
    'Make system sound quieter', 'Raise audio volume slightly',
    'اخفض مستوى الصوت للنص', 'خلي الصوت على ٤٠ في المية',
    'ارفع صوت الجهاز لأقصى درجة', 'خلي الصوت على أقل درجة',
    'رجع الصوت الطبيعي', 'عايز اقلل صوت الجهاز',
],
'LOCK_SCREEN': [
    'Lock my workstation now', 'Secure the current desktop session',
    'Please lock this computer', 'Lock the current user session',
    'Put the computer screen on lock', 'Lock my Ubuntu session',
    'Secure my desktop', 'Lock the laptop screen', 'Please lock the PC',
    'عايز اقفل الشاشة مؤقتا', 'اعمل قفل للجهاز', 'أمّن جلسة المستخدم',
    'اقفل السيشن بتاعتي', 'ممكن تقفل شاشة اللابتوب', 'خلي الجهاز على شاشة القفل',
    'اقفل سطح المكتب مؤقتا', 'اعمل لوك للكمبيوتر',
    'Lock this user session', 'Please secure my desktop session',
    'Put my workstation into locked mode', 'Lock the current laptop session',
    'Lock the computer before I leave', 'Secure this desktop now',
    'ممكن تعمل قفل لسطح المكتب', 'اقفل جلسة اوبونتو',
    'حط الجهاز على شاشة القفل', 'عايز اقفل السيشن الحالية',
],
'SHUTDOWN': [
    'Power down the computer', 'Please turn off Ubuntu', 'Shut the laptop down safely',
    'Initiate system shutdown', 'I want to power off the PC', 'Turn the computer completely off',
    'Shut down this machine', 'Please shut the system down', 'Switch the PC off',
    'اطفي اوبونتو', 'ممكن تعمل إيقاف تشغيل للجهاز', 'عايز اقفل الكمبيوتر نهائي',
    'اطفي اللابتوب خالص', 'ابدأ إغلاق النظام', 'عايز أوقف تشغيل الكمبيوتر',
    'اقفل الجهاز إيقاف تشغيل', 'نفذ إيقاف تشغيل اوبونتو',
    'Power the machine down completely', 'Please switch off the laptop now',
    'Shut down my workstation when ready', 'Turn this computer off now',
    'أوقف تشغيل الجهاز بالكامل', 'ممكن تطفي اللابتوب', 'اقفل اوبونتو بشكل كامل',
],
'CREATE_FOLDER': [
    'Create a directory called Semester 3', 'Make a folder named Research',
    'Create a new directory for my projects', 'Make a folder called Statistics',
    'Please create a folder named Portfolio', 'Add a new directory named Archive',
    'Create a folder in Documents called Notes', 'Make a directory named Data Analysis',
    'Create a new folder for invoices', 'Make a directory for university work',
    'ممكن تعمل مجلد اسمه الجامعة', 'أنشئ فولدر باسم المشاريع', 'اعمل مجلد جديد اسمه التقارير',
    'اعمل فولدر باسم الاحصاء', 'أنشئ مجلد للمحاضرات', 'اعمل مجلد اسمه الداتا',
    'محتاج فولدر جديد اسمه السيرة الذاتية', 'اعمل مجلد باسم الأرشيف',
],
'OPEN_YOUTUBE': [
    'Could you open YouTube for me?', 'Please take me to the YouTube homepage',
    'Open the YouTube website in my browser', 'Go to the YouTube home page',
    'Launch YouTube in Firefox', 'Visit YouTube homepage', 'Open YouTube home',
    'ممكن تفتح يوتيوب', 'ممكن تفتحلي اليوتيوب', 'عايز تفتح يوتيوب',
    'ممكن تدخل على يوتيوب', 'افتحلي موقع اليوتيوب', 'لو سمحت افتح يوتيوب',
    'عايزة افتح الصفحة الرئيسية ليوتيوب', 'روح لليوتيوب', 'ممكن توديني على يوتيوب',
    'Bring up the YouTube page', 'Navigate to the YouTube homepage',
    'Show me the YouTube homepage', 'Load the YouTube website',
    'Can you take me to the YouTube page?', 'I would like to visit YouTube',
    'Please navigate to youtube.com', 'Open youtube.com in the browser',
    'Go to the main YouTube page', 'Take me to the YouTube website',
    'Launch the YouTube homepage', 'I want to visit YouTube',
    'ممكن تفتح موقع اليوتيوب عندي', 'افتح صفحة يوتيوب الرئيسية',
    'عايز اروح على موقع يوتيوب', 'ادخل على صفحة اليوتيوب',
    'ممكن توديني للموقع بتاع يوتيوب', 'افتح اليوتيوب في المتصفح',
    'شغل موقع يوتيوب الرئيسي', 'روح على صفحة يوتيوب',
    'عايزة ادخل على يوتيوب', 'افتح موقع youtube.com',
    'Access the YouTube website', 'Open YouTube in a web browser',
    'Open the YouTube home screen', 'Display the YouTube homepage',
    'Take me straight to youtube.com', 'Bring YouTube up in Firefox',
    'Could you navigate to the YouTube site?', 'I need the YouTube homepage',
    'Please access YouTube from my browser', 'Visit the main YouTube page',
    'Open the homepage for YouTube', 'Start the YouTube website',
    'Load youtube.com for me', 'Show the main page on YouTube',
    'Can you bring up youtube.com?', 'Navigate my browser to YouTube',
    'افتح يوتيوب على المتصفح الافتراضي', 'عايز ادخل على موقع يوتيوب',
    'ممكن تفتح صفحة اليوتيوب عندي', 'وديني لصفحة اليوتيوب الرئيسية',
    'افتح يوتيوب في فايرفوكس', 'محتاج افتح موقع يوتيوب',
    'ممكن تفتحلي موقع youtube.com', 'ادخل على الصفحة الرئيسية لليوتيوب',
    'عايز اشوف الصفحة الرئيسية ليوتيوب', 'افتح موقع اليوتيوب في المتصفح',
],
'PLAY_MEDIA': [
    'Play Interstellar', 'Play the movie Inception', 'Play Bohemian Rhapsody',
    'Play the song Tamally Maak', 'Play my downloaded movie', 'Play the local video file',
    'Play the music file from my Music folder', 'Play a movie from my local library',
    'Open the song in my music collection', 'Play the track called Believer',
    'Play a video from my Movies folder', 'Start playing the local audio file',
    'Play the film saved on my computer', 'Play the song in my local library',
    'Play the file named lecture.mp4', 'Play my downloaded music',
    'شغل فيلم انترستيلر', 'شغل فيلم انسبشن', 'شغل أغنية تملي معاك',
    'شغللي أغنية نور العين', 'شغل الفيديو الموجود عندي', 'شغل الفيلم من مجلد الأفلام',
    'شغل الأغنية من مجلد الموسيقى', 'شغل الملف الصوتي المحلي',
    'شغل الفيديو اللي على الجهاز', 'شغل فيلم من مكتبتي', 'شغل المقطع المحفوظ عندي',
    'شغل أغنية عمرو دياب المحفوظة', 'افتح ملف الفيديو وشغله',
    'ممكن تشغل الفيلم الموجود في الجهاز', 'شغل الملف interstellar.mkv',
],
'UNKNOWN': [
    'Tell me a joke', 'Write a poem about the sea', 'What is the weather today?',
    'Translate this paragraph into French', 'Calculate my university GPA',
    'Send an email to my professor', 'Summarize this research paper',
    'Who won yesterday’s football match?', 'Explain quantum mechanics to me',
    'What is the capital of France?', 'Create a presentation about climate change',
    'Book a table for dinner', 'Remind me about tomorrow’s meeting',
    'Generate an image of a cat', 'What time is it in Tokyo?',
    'ابعت ايميل للدكتور', 'احكيلي نكتة', 'اكتبلي قصيدة عن البحر',
    'ترجم الفقرة دي للفرنساوي', 'احسبلي المعدل التراكمي', 'ايه حالة الطقس النهارده؟',
    'مين كسب ماتش امبارح؟', 'لخص البحث العلمي ده', 'اشرحلي ميكانيكا الكم',
    'احجزلي ترابيزة للعشاء', 'فكرني باجتماع بكرة', 'اعمل صورة لقطة',
    'ايه عاصمة فرنسا؟', 'اكتبلي موضوع تعبير عن التغير المناخي',
],
}

# Additional controlled templates for broader natural phrasing.
APP_EN = ['Firefox', 'Visual Studio Code', 'the terminal', 'the calculator', 'the file manager', 'the text editor', 'the settings app', 'the system monitor', 'the document viewer', 'the browser']
APP_AR = ['فايرفوكس', 'في اس كود', 'التيرمنال', 'الآلة الحاسبة', 'مدير الملفات', 'محرر النصوص', 'الإعدادات', 'مراقب النظام']
FOLDER_EN = ['Downloads', 'Documents', 'Desktop', 'Music', 'Videos', 'Pictures', 'Projects', 'Assignments', 'Reports', 'Archive']
FOLDER_AR = ['التنزيلات', 'المستندات', 'سطح المكتب', 'الموسيقى', 'الفيديوهات', 'الصور', 'المشاريع', 'الواجبات', 'التقارير']
FILES_EN = ['report.pdf', 'notes.txt', 'budget.xlsx', 'project presentation', 'my CV', 'the assignment PDF', 'statistics notes', 'invoice.docx', 'dashboard image', 'lecture file']
FILES_AR = ['ملف التقرير', 'ملف الواجب', 'ملف الإكسل', 'ملف المحاضرة', 'ملف السيرة الذاتية', 'ملف المشروع', 'صورة الداشبورد', 'ملف الفاتورة']

# Label-specific generated examples. These are intentionally limited to command
# forms whose expected intent is unambiguous.
def generate_for(intent):
    out = []
    if intent == 'OPEN_APP':
        for app in APP_EN:
            for frame in ['Open {x}', 'Launch {x}', 'Please open {x}', 'Could you launch {x}?', 'Start {x} for me', 'I need to open {x}']:
                out.append(frame.format(x=app))
        for app in APP_AR:
            for frame in ['افتح {x}', 'ممكن تفتح {x}', 'شغل {x}', 'عايز افتح {x}', 'افتحلي {x}', 'لو سمحت افتح {x}']:
                out.append(frame.format(x=app))
    elif intent == 'CLOSE_APP':
        for app in APP_EN:
            for frame in ['Close {x}', 'Quit {x}', 'Please close {x}', 'Could you quit {x}?', 'Exit {x} for me', 'I need to close {x}']:
                out.append(frame.format(x=app))
        for app in APP_AR:
            for frame in ['اقفل {x}', 'ممكن تقفل {x}', 'اغلق {x}', 'اقفللي {x}', 'عايز اقفل {x}', 'لو سمحت اقفل {x}']:
                out.append(frame.format(x=app))
    elif intent == 'OPEN_FOLDER':
        for folder in FOLDER_EN:
            for frame in ['Open {x}', 'Open the {x} folder', 'Show me {x}', 'Go to the {x} folder', 'Take me to {x}', 'Please open the folder {x}']:
                out.append(frame.format(x=folder))
        for folder in FOLDER_AR:
            for frame in ['افتح {x}', 'افتح مجلد {x}', 'وريني {x}', 'روح على مجلد {x}', 'ممكن تفتح {x}']:
                out.append(frame.format(x=folder))
    elif intent == 'SEARCH_FILE':
        for file in FILES_EN:
            for frame in ['Find {x}', 'Search for {x}', 'Locate {x}', 'Look for {x}', 'Can you find {x}?', 'Where is {x}?']:
                out.append(frame.format(x=file))
        for file in FILES_AR:
            for frame in ['دور على {x}', 'ابحث عن {x}', 'فين {x}', 'ممكن تلاقي {x}', 'هات {x}']:
                out.append(frame.format(x=file))
    elif intent == 'GET_CPU':
        terms = ['CPU usage', 'processor utilization', 'CPU load', 'processor activity', 'CPU percentage', 'معالج الجهاز', 'استهلاك المعالج', 'البروسيسور', 'السي بي يو']
        for term in terms:
            for frame in ['Check {x}', 'Show {x}', 'Tell me the {x}', 'Display current {x}', 'What is the current {x}?', 'اعرض {x}', 'هات {x}', '{x} كام؟']:
                out.append(frame.format(x=term))
    elif intent == 'GET_RAM':
        terms = ['RAM usage', 'memory utilization', 'available RAM', 'free memory', 'used memory', 'الرام', 'استهلاك الرام', 'الذاكرة المستخدمة', 'الذاكرة المتاحة']
        for term in terms:
            for frame in ['Check {x}', 'Show {x}', 'Tell me the {x}', 'Display current {x}', 'How much {x} is left?', 'اعرض {x}', 'هات {x}', '{x} كام؟']:
                out.append(frame.format(x=term))
    elif intent == 'GET_DISK':
        terms = ['disk space', 'free storage', 'hard drive space', 'available capacity', 'remaining storage', 'مساحة الهارد', 'مساحة التخزين', 'المساحة الفاضية', 'مساحة القرص']
        for term in terms:
            for frame in ['Check {x}', 'Show {x}', 'Tell me the {x}', 'Display current {x}', 'How much {x} remains?', 'اعرض {x}', 'هات {x}', '{x} كام؟']:
                out.append(frame.format(x=term))
    elif intent == 'CHANGE_VOLUME':
        levels = ['10', '20', '30', '50', '70', '80', '100']
        for level in levels:
            for frame in ['Set volume to {x} percent', 'Set the sound level to {x}', 'Make the volume {x} percent', 'خلي الصوت {x} في المية']:
                out.append(frame.format(x=level))
        for verb in ['increase', 'raise', 'turn up', 'boost', 'lower', 'decrease', 'reduce', 'mute', 'unmute']:
            out.append(f'{verb.capitalize()} the volume')
        for verb in ['ارفع', 'زود', 'علي', 'وطي', 'قلل', 'اكتم', 'افتح']:
            out.append(f'{verb} الصوت')
    elif intent == 'LOCK_SCREEN':
        targets = ['screen', 'desktop', 'workstation', 'PC screen', 'computer session', 'Ubuntu session', 'شاشة الكمبيوتر', 'الشاشة', 'جلسة المستخدم', 'سطح المكتب']
        for target in targets:
            for frame in ['Lock the {x}', 'Please lock the {x}', 'Secure the {x}', 'اقفل {x}', 'اعمل قفل لـ{x}']:
                out.append(frame.format(x=target))
    elif intent == 'SHUTDOWN':
        targets = ['computer', 'PC', 'laptop', 'Ubuntu', 'system', 'workstation', 'الكمبيوتر', 'اللابتوب', 'الجهاز', 'اوبونتو']
        for target in targets:
            for frame in ['Shut down the {x}', 'Power off the {x}', 'Turn off the {x}', 'Please shut down the {x}', 'اطفي {x}', 'اقفل {x} إيقاف تشغيل']:
                out.append(frame.format(x=target))
    elif intent == 'CREATE_FOLDER':
        names = ['Projects', 'Reports', 'Assignments', 'Semester 3', 'Data Analysis', 'Research', 'Portfolio', 'Statistics', 'University', 'Archive', 'المشاريع', 'التقارير', 'الواجبات', 'الجامعة', 'الاحصاء', 'المحاضرات', 'الشغل', 'الأرشيف']
        for name in names:
            for frame in ['Create a folder named {x}', 'Make a folder called {x}', 'Create directory {x}', 'Make a new directory named {x}', 'اعمل فولدر اسمه {x}', 'أنشئ مجلد باسم {x}']:
                out.append(frame.format(x=name))
    elif intent == 'OPEN_YOUTUBE':
        for frame in ['Open YouTube', 'Open YouTube for me', 'Please open YouTube', 'Go to YouTube', 'Take me to YouTube', 'Open the YouTube website', 'Could you open YouTube?', 'ممكن تفتح يوتيوب', 'ممكن تفتحلي اليوتيوب', 'افتح موقع يوتيوب', 'عايز افتح يوتيوب', 'ادخل على اليوتيوب', 'روح على يوتيوب']:
            out.append(frame)
    elif intent == 'PLAY_MEDIA':
        titles = ['Interstellar', 'Inception', 'The Godfather', 'Believer', 'Tamally Maak', 'Nour El Ein', 'Bohemian Rhapsody', 'lecture.mp4', 'family_video.mp4', 'my downloaded song', 'music from my library', 'the movie in Downloads', 'انترستيلر', 'انسبشن', 'تملي معاك', 'نور العين', 'فيديو المحاضرة', 'فيلم الكنز', 'الأغنية المحفوظة', 'فيديو العائلة']
        for title in titles:
            for frame in ['Play {x}', 'Play the song {x}', 'Play the movie {x}', 'Play {x} from my local library', 'شغل {x}', 'شغل أغنية {x}', 'شغل فيلم {x}']:
                out.append(frame.format(x=title))
    elif intent == 'UNKNOWN':
        topics = ['the weather', 'a joke', 'a poem', 'my GPA', 'an email', 'a restaurant reservation', 'a translation', 'a research summary', 'tomorrow’s weather', 'a PowerPoint presentation', 'the stock market', 'a reminder', 'a math problem', 'the football score', 'a bedtime story', 'الطقس', 'نكتة', 'قصيدة', 'المعدل التراكمي', 'ايميل للدكتور', 'ترجمة النص', 'ملخص البحث', 'حجز مطعم', 'نتيجة الماتش', 'قصة قبل النوم']
        frames = ['Tell me about {x}', 'Can you write {x}?', 'I want you to create {x}', 'Please explain {x}', 'اريد {x}', 'ممكن تعمل {x}', 'احكيلي عن {x}']
        for topic in topics:
            for frame in frames:
                out.append(frame.format(x=topic))
    return out

# Existing explicit YouTube search topics are already numerous; don't flood them
# with templated duplicates. PLAY_MEDIA is now its own intent per the project plan.
base['normalized'] = base['text'].map(norm)
# Remove identical normalized phrases, keeping the first row.
base = base.drop_duplicates(subset=['normalized'], keep='first')
rows = [{'text': r.text, 'intent': r.intent} for r in base.itertuples()]

# Exclude phrases that are likely to be routed elsewhere by the parser from class generation.
TARGET_COUNT = 90
all_classes = [
    'CHANGE_VOLUME','CLOSE_APP','CREATE_FOLDER','GET_CPU','GET_DISK','GET_RAM',
    'LOCK_SCREEN','OPEN_APP','OPEN_FOLDER','OPEN_YOUTUBE','PLAY_MEDIA',
    'SEARCH_FILE','SEARCH_YOUTUBE','SHUTDOWN','UNKNOWN'
]

# Global normalized map ensures we never label the same normalized phrase differently.
label_by_norm = {}
for row in rows:
    key = norm(row['text'])
    if key not in label_by_norm:
        label_by_norm[key] = row['intent']

for intent in all_classes:
    # Add hand-curated targeted phrases first.
    candidates = list(SUPPLEMENT.get(intent, []))
    # Add controlled variations; filters below remove duplicates and conflicts.
    candidates += generate_for(intent)
    current = sum(1 for row in rows if row['intent'] == intent)
    # Continue using generated candidates until the per-class target is reached.
    for phrase in candidates:
        if current >= TARGET_COUNT and intent != 'SEARCH_YOUTUBE':
            break
        if intent == 'SEARCH_YOUTUBE' and current >= 110:
            break
        phrase = str(phrase).strip()
        if not phrase:
            continue
        key = norm(phrase)
        if key in label_by_norm:
            if label_by_norm[key] == intent:
                continue
            else:
                continue
        rows.append({'text': phrase, 'intent': intent})
        label_by_norm[key] = intent
        current += 1

# Keep baseline bare playback examples out of SEARCH_YOUTUBE. Any rows that got left
# from the base but are still bare play/song commands are relabeled carefully.
for row in rows:
    if row['intent'] == 'SEARCH_YOUTUBE' and should_move_to_media(row['text']):
        row['intent'] = 'PLAY_MEDIA'

# Deduplicate and check conflicts after relabeling.
df = pd.DataFrame(rows)
df['normalized'] = df['text'].map(norm)
# If normalized duplicates now disagree, retain intent of first (curated/baseline) row.
df = df.drop_duplicates(subset=['normalized'], keep='first').drop(columns='normalized')
# Ensure every intended intent exists; print counts.
counts = df['intent'].value_counts().to_dict()
assert 'PLAY_MEDIA' in counts
assert len(counts) == 15
# Class balance cap SEARCH_YOUTUBE set not applied if original 103; okay.
df.to_csv(base_path, index=False, encoding='utf-8-sig')
df.to_csv(OUT / 'training_examples_v6.csv', index=False, encoding='utf-8-sig')
# Keep the original v5 dataset backup unchanged.

# Independent, explicitly held-out evaluation phrases: do NOT add these to training data.
EVAL = [
('Please bring up the browser', 'OPEN_APP'),
('Could you start the calculator application?', 'OPEN_APP'),
('ممكن تفتح مدير الملفات', 'OPEN_APP'),
('افتح برنامج عرض الصور', 'OPEN_APP'),
('Please exit the application', 'CLOSE_APP'),
('ممكن تقفل فايرفوكس من فضلك', 'CLOSE_APP'),
('Quit the text editor for me', 'CLOSE_APP'),
('وريني مجلد المستندات', 'OPEN_FOLDER'),
('Take me to the Pictures folder', 'OPEN_FOLDER'),
('افتح فولدر التنزيلات', 'OPEN_FOLDER'),
('Could you locate the thesis document?', 'SEARCH_FILE'),
('Look up the annual report file', 'SEARCH_FILE'),
('ممكن تدورلي على ملف الورد', 'SEARCH_FILE'),
('فين ملف الداشبورد', 'SEARCH_FILE'),
('What is the current processor load?', 'GET_CPU'),
('اعرض ضغط المعالج', 'GET_CPU'),
('Show available memory', 'GET_RAM'),
('الرام المتاحة كام', 'GET_RAM'),
('How much space is unused on the drive?', 'GET_DISK'),
('فاضل كام مساحة على القرص', 'GET_DISK'),
('Please reduce the sound level', 'CHANGE_VOLUME'),
('اكتم صوت السماعة', 'CHANGE_VOLUME'),
('Secure my current screen', 'LOCK_SCREEN'),
('ممكن تعمل قفل للسيشن', 'LOCK_SCREEN'),
('Turn the laptop off', 'SHUTDOWN'),
('اطفي الجهاز تماما دلوقتي', 'SHUTDOWN'),
('Create a folder called Semester 4', 'CREATE_FOLDER'),
('اعمل مجلد باسم التخرج', 'CREATE_FOLDER'),
('Would you open YouTube for me?', 'OPEN_YOUTUBE'),
('ممكن توديني لصفحة اليوتيوب', 'OPEN_YOUTUBE'),
('Play the movie Interstellar from my collection', 'PLAY_MEDIA'),
('شغل الفيلم المحفوظ عندي', 'PLAY_MEDIA'),
('Search YouTube for a tutorial on SQL window functions', 'SEARCH_YOUTUBE'),
('ابحث على يوتيوب عن شرح الاحتمال الشرطي', 'SEARCH_YOUTUBE'),
('Find my notes file', 'SEARCH_FILE'),
('What is the weather tomorrow?', 'UNKNOWN'),
('اكتبلي قصة قصيرة', 'UNKNOWN'),
('Play the song Tamally Maak from my local music folder', 'PLAY_MEDIA'),
('Search YouTube for rock and roll guitar lessons', 'SEARCH_YOUTUBE'),
('Open YouTube and search for Markov chains', 'SEARCH_YOUTUBE'),
]
eval_df = pd.DataFrame(EVAL, columns=['text','intent'])
eval_df.to_csv(OUT / 'independent_eval_v6.csv', index=False, encoding='utf-8-sig')

# Add a data manifest.
manifest = {
  'project': 'Voice Ubuntu NLP',
  'version': 'v6',
  'source_dataset_rows_v5': int(len(pd.read_csv(OUT / 'training_examples_v5_original.csv'))),
  'expanded_dataset_rows': int(len(df)),
  'intent_count': int(df.intent.nunique()),
  'intent_counts': {str(k): int(v) for k,v in df.intent.value_counts().sort_index().items()},
  'independent_eval_examples': int(len(eval_df)),
  'independent_eval_policy': 'independent_eval_v6.csv is kept separate and never used for model training',
  'dataset_note': 'Expanded data includes curated bilingual phrases and carefully generated command template variants. Metrics must be interpreted as preliminary; independent real user transcripts are still needed.',
}
(OUT / 'dataset_manifest_v6.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print('DATASET ROWS', len(df), 'INTENTS', df.intent.nunique())
print(df.intent.value_counts().sort_index().to_string())
print('Independent eval:', len(eval_df))
