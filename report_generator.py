"""
Cardio Digital Twin — Professional Medical Report Generator
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)

# PALETTE CLINIQUE
TEAL        = colors.HexColor('#0d9488')
TEAL_DARK   = colors.HexColor('#0f766e')
TEAL_LIGHT  = colors.HexColor('#f0fdfa')
NAVY        = colors.HexColor('#0c2340')
NAVY_LIGHT  = colors.HexColor('#0e3a5e')
EMERALD     = colors.HexColor('#059669')
AMBER       = colors.HexColor('#d97706')
ROSE        = colors.HexColor('#e11d48')
GRAY_100    = colors.HexColor('#f8fffe')
GRAY_200    = colors.HexColor('#e6fdf8')
GRAY_400    = colors.HexColor('#4d9e96')
GRAY_600    = colors.HexColor('#1e5a52')
WHITE       = colors.white

# STYLES
def get_styles():
    return {
        'Title': ParagraphStyle('Title', fontName='Helvetica-Bold',
            fontSize=24, leading=30, textColor=WHITE, alignment=TA_CENTER),
        'Subtitle': ParagraphStyle('Subtitle', fontName='Helvetica',
            fontSize=12, leading=16, textColor=colors.HexColor('#99f6e4'), alignment=TA_CENTER),
        'SectionTitle': ParagraphStyle('SectionTitle', fontName='Helvetica-Bold',
            fontSize=13, leading=18, textColor=TEAL_DARK, spaceBefore=16, spaceAfter=8),
        'Body': ParagraphStyle('Body', fontName='Helvetica',
            fontSize=10, leading=15, textColor=GRAY_600, spaceAfter=4),
        'Small': ParagraphStyle('Small', fontName='Helvetica',
            fontSize=8, leading=12, textColor=GRAY_400),
        'Conclusion': ParagraphStyle('Conclusion', fontName='Helvetica',
            fontSize=10, leading=14, textColor=GRAY_600, spaceAfter=6),
        'PatientName': ParagraphStyle('PatientName', fontName='Helvetica-Bold',
            fontSize=14, leading=18, textColor=NAVY, alignment=TA_CENTER),
    }

# HEADER / FOOTER
class MedicalCanvas:
    def __init__(self, patient_name, report_date):
        # Nettoyer le nom du patient
        if not patient_name or patient_name == '' or patient_name == 'None' or patient_name == '(anonymous)' or patient_name == 'anonymous':
            patient_name = 'Patient'
        self.patient_name = patient_name
        self.report_date = report_date

    def __call__(self, canvas, doc):
        canvas.saveState()
        w, h = A4

        canvas.setFillColor(NAVY)
        canvas.rect(0, h - 60, w, 60, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, h - 63, w, 3, fill=1, stroke=0)

        canvas.setFont('Helvetica-Bold', 15)
        canvas.setFillColor(WHITE)
        canvas.drawString(24, h - 36, 'CARDIO DIGITAL TWIN')
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#99f6e4'))
        canvas.drawString(24, h - 50, "Systeme d'aide a la decision cardiovasculaire")
        canvas.drawRightString(w - 24, h - 30, f'Patient : {self.patient_name}')
        canvas.drawRightString(w - 24, h - 44, f'Date : {self.report_date}')
        canvas.drawRightString(w - 24, h - 57, f'Page {doc.page}')

        canvas.setFillColor(GRAY_200)
        canvas.rect(0, 0, w, 36, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.rect(0, 36, w, 1, fill=1, stroke=0)
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(GRAY_400)
        canvas.drawCentredString(w/2, 22, 'Document genere automatiquement par Cardio Digital Twin')
        canvas.drawCentredString(w/2, 12, "Il ne remplace pas un avis medical professionnel")
        canvas.restoreState()

# HELPERS
def section_header(title, bg=None):
    bg = bg or TEAL
    s = get_styles()
    data = [[Paragraph(f'<font color="white"><b>{title}</b></font>',
             ParagraphStyle('H', fontName='Helvetica-Bold', fontSize=12,
                            leading=16, textColor=WHITE))]]
    t = Table(data, colWidths=[16.2*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
    ]))
    return t

def info_table(rows, col_widths=None):
    col_widths = col_widths or [6*cm, 10.2*cm]
    s = get_styles()
    data = []
    for label, value in rows:
        data.append([
            Paragraph(f'<b>{label}</b>', s['Body']),
            Paragraph(str(value), s['Body']),
        ])
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [GRAY_100, WHITE]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#ccfbf1')),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (0,-1), NAVY),
        ('TEXTCOLOR', (1,0), (1,-1), GRAY_600),
    ]))
    return t

# HELPERS UCI
def cp_label(cp):
    return {1:'Angine typique', 2:'Angine atypique',
            3:'Douleur non angineuse', 4:'Asymptomatique'}.get(int(cp), str(cp))

def sex_label(sex):
    return 'Homme' if int(sex) == 1 else 'Femme'

def restecg_label(r):
    return {0:'Normal', 1:'Anomalie onde ST-T',
            2:'Hypertrophie ventriculaire gauche'}.get(int(r), str(r))

def slope_label(s):
    return {1:'Montante', 2:'Plate', 3:'Descendante'}.get(int(s), str(s))

def thal_label(t):
    return {3:'Normal', 6:'Defaut fixe', 7:'Defaut reversible'}.get(int(t), str(t))

def chol_status(chol):
    c = float(chol)
    if c > 240: return f'{c:.0f} mg/dL — Tres eleve'
    if c > 200: return f'{c:.0f} mg/dL — Eleve'
    return f'{c:.0f} mg/dL — Normal'

def bp_status(bp):
    b = float(bp)
    if b > 140: return f'{b:.0f} mmHg — Hypertension'
    if b > 130: return f'{b:.0f} mmHg — Eleve'
    return f'{b:.0f} mmHg — Normal'

# MAIN CLASS
class ReportGenerator:

    def generate_pdf(self, patient_data, result, recommendations, filename=None):
        """Rapport simple - une seule analyse"""
        if filename is None:
            filename = f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        reports_dir = os.path.join('static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        filepath = os.path.join(reports_dir, filename)

        s = get_styles()
        patient_name = patient_data.get('name', 'Patient')
        # Nettoyer le nom
        if not patient_name or patient_name == '' or patient_name == 'None' or patient_name == '(anonymous)':
            patient_name = 'Patient'
        report_date = datetime.now().strftime('%d/%m/%Y a %H:%M')
        footer = MedicalCanvas(patient_name, report_date)

        # AJOUT DES MÉTADONNÉES PDF
        doc = SimpleDocTemplate(filepath, pagesize=A4,
            topMargin=72, bottomMargin=52, leftMargin=24, rightMargin=24,
            title=f"Rapport medical - {patient_name}",
            author="Cardio Digital Twin",
            subject=f"Analyse de risque cardiovasculaire")

        story = []
        prob = result.get('probability', 0)

        story.append(Spacer(1, 1.5*cm))
        cover = Table([[Paragraph('RAPPORT D\'ANALYSE DE RISQUE', s['Title'])]], colWidths=[16.2*cm])
        cover.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), TEAL),
            ('TOPPADDING', (0,0), (-1,-1), 40),
            ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ]))
        story.append(cover)
        
        sub = Table([[Paragraph(f'Patient : {patient_name}', s['Subtitle'])]], colWidths=[16.2*cm])
        sub.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), TEAL_DARK),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ]))
        story.append(sub)
        story.append(Spacer(1, 1*cm))

        color = ROSE if prob > 0.6 else (AMBER if prob > 0.3 else EMERALD)
        level = 'ELEVE' if prob > 0.6 else ('MODERE' if prob > 0.3 else 'FAIBLE')

        risk_t = Table([[
            Paragraph(f'<font color="white"><b>RISQUE : {level}</b></font>',
                      ParagraphStyle('rr', fontName='Helvetica-Bold', fontSize=14,
                                     leading=18, textColor=WHITE)),
            Paragraph(f'<font color="white"><b>{prob*100:.1f}%</b></font>',
                      ParagraphStyle('rp', fontName='Helvetica-Bold', fontSize=28,
                                     leading=34, textColor=WHITE, alignment=TA_RIGHT)),
        ]], colWidths=[10*cm, 6.2*cm])
        risk_t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), color),
            ('TOPPADDING', (0,0), (-1,-1), 15),
            ('BOTTOMPADDING', (0,0), (-1,-1), 15),
        ]))
        story.append(risk_t)
        story.append(Spacer(1, 6*mm))

        params = [
            ('Age', f"{patient_data.get('age','—')} ans"),
            ('Sexe', sex_label(patient_data.get('sex', 1))),
            ('Type douleur', cp_label(patient_data.get('cp', 4))),
            ('Tension', bp_status(patient_data.get('trestbps', 120))),
            ('Cholesterol', chol_status(patient_data.get('chol', 200))),
            ('FC maximale', f"{patient_data.get('thalach','—')} bpm"),
            ("Angine d'effort", 'Oui' if patient_data.get('exang') == 1 else 'Non'),
        ]
        story.append(info_table(params))

        story.append(Spacer(1, 6*mm))
        story.append(Paragraph(
            "Ce rapport a ete genere automatiquement par Cardio Digital Twin.",
            s['Small']
        ))

        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        return filepath

    def generate_full_report(self, patient_name, consultations, filename=None):
        """Rapport complet - une consultation par page avec nom du patient"""
        

        # NETTOYAGE DU NOM DU PATIENT
        if not patient_name or patient_name == '' or patient_name == 'None' or patient_name == '(anonymous)' or patient_name == 'anonymous':
            # Essayer de récupérer le nom depuis la première consultation
            if consultations and len(consultations) > 0:
                patient_name = consultations[0].get('patient_name', 'Patient')
            else:
                patient_name = 'Patient'
        
        patient_name = patient_name.strip()
        
        if filename is None:
            filename = f"rapport_{patient_name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        reports_dir = os.path.join('static', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        filepath = os.path.join(reports_dir, filename)

        s = get_styles()
        report_date = datetime.now().strftime('%d/%m/%Y a %H:%M')
        footer = MedicalCanvas(patient_name, report_date)

        
        # AJOUT DES MÉTADONNÉES PDF 

        doc = SimpleDocTemplate(filepath, pagesize=A4,
            topMargin=72, bottomMargin=52, leftMargin=24, rightMargin=24,
            title=f"Rapport medical complet - {patient_name}",
            author="Cardio Digital Twin",
            subject=f"Analyses cardiovasculaires de {patient_name}")

        story = []


        # PAGE DE GARDE

        story.append(Spacer(1, 2*cm))
        cover = Table([[Paragraph('RAPPORT MEDICAL COMPLET', s['Title'])]], colWidths=[16.2*cm])
        cover.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), NAVY),
            ('TOPPADDING', (0,0), (-1,-1), 40),
            ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ]))
        story.append(cover)
        
        sub = Table([[Paragraph(f'Patient : {patient_name}', s['Subtitle'])]], colWidths=[16.2*cm])
        sub.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), NAVY_LIGHT),
            ('TOPPADDING', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 30),
        ]))
        story.append(sub)
        story.append(Spacer(1, 1*cm))
        
        story.append(Paragraph(f"<b>Date du rapport :</b> {report_date}", s['Body']))
        story.append(Paragraph(f"<b>Nombre total de consultations :</b> {len(consultations)}", s['Body']))
        
        # Résumé des consultations par type
        risk_count = len([c for c in consultations if c.get('type') == 'risk'])
        ecg_count = len([c for c in consultations if c.get('type') == 'ecg'])
        xray_count = len([c for c in consultations if c.get('type') == 'xray'])
        
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph("<b>Résumé des analyses :</b>", s['Body']))
        story.append(Paragraph(f"• Analyses de risque : {risk_count}", s['Body']))
        story.append(Paragraph(f"• Analyses ECG : {ecg_count}", s['Body']))
        story.append(Paragraph(f"• Radiographies : {xray_count}", s['Body']))

        # SÉPARATION DES CONSULTATIONS PAR TYPE

        risk_consults = [c for c in consultations if c.get('type') == 'risk' and 'prediction' in c]
        ecg_consults = [c for c in consultations if c.get('type') == 'ecg' and 'result' in c]
        xray_consults = [c for c in consultations if c.get('type') == 'xray' and 'result' in c]
        
        risk_consults.sort(key=lambda x: x.get('date', ''))
        ecg_consults.sort(key=lambda x: x.get('date', ''))
        xray_consults.sort(key=lambda x: x.get('date', ''))

        # 1. ANALYSES DE RISQUE 

        for i, c in enumerate(risk_consults):
            story.append(PageBreak())
            
            pred = c.get('prediction', {})
            p = c.get('patient', {})
            prob = pred.get('probability', 0)
            date_str = c['date'][:10]
            time_str = c['date'][11:16] if len(c['date']) > 10 else ''

            # En-tête de la page
            story.append(section_header(f'ANALYSE DE RISQUE N°{i+1}'))
            story.append(Spacer(1, 4*mm))
            
            # Identité patient et date
            story.append(Paragraph(f"<b>Patient :</b> {patient_name}", s['Body']))
            story.append(Paragraph(f"<b>Date de l'analyse :</b> {date_str} à {time_str}", s['Body']))
            story.append(Spacer(1, 6*mm))

            # Badge de risque
            color = ROSE if prob > 0.6 else (AMBER if prob > 0.3 else EMERALD)
            level = 'ELEVE' if prob > 0.6 else ('MODERE' if prob > 0.3 else 'FAIBLE')

            risk_t = Table([[
                Paragraph(f'<font color="white"><b>NIVEAU DE RISQUE : {level}</b></font>',
                          ParagraphStyle('rr', fontName='Helvetica-Bold', fontSize=12,
                                         leading=16, textColor=WHITE)),
                Paragraph(f'<font color="white"><b>{prob*100:.1f}%</b></font>',
                          ParagraphStyle('rp', fontName='Helvetica-Bold', fontSize=24,
                                         leading=30, textColor=WHITE, alignment=TA_RIGHT)),
            ]], colWidths=[11*cm, 5.2*cm])
            risk_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), color),
                ('TOPPADDING', (0,0), (-1,-1), 12),
                ('BOTTOMPADDING', (0,0), (-1,-1), 12),
                ('LEFTPADDING', (0,0), (-1,-1), 14),
                ('RIGHTPADDING', (0,0), (-1,-1), 14),
            ]))
            story.append(risk_t)
            story.append(Spacer(1, 6*mm))

            # Paramètres UCI
            params = [
                ('Âge', f"{p.get('age','—')} ans"),
                ('Sexe', sex_label(p.get('sex', 1))),
                ('Type douleur thoracique', cp_label(p.get('cp', 4))),
                ('Tension artérielle', bp_status(p.get('trestbps', 120))),
                ('Cholestérol', chol_status(p.get('chol', 200))),
                ('Glycémie à jeun', 'Oui' if p.get('fbs') == 1 else 'Non'),
                ('ECG au repos', restecg_label(p.get('restecg', 0))),
                ('FC maximale', f"{p.get('thalach','—')} bpm"),
                ("Angine d'effort", 'Oui' if p.get('exang') == 1 else 'Non'),
                ('Dépression ST (oldpeak)', str(p.get('oldpeak', 0))),
                ('Pente du segment ST', slope_label(p.get('slope', 1))),
                ('Vaisseaux colorés (ca)', str(p.get('ca', 0))),
                ('Thalassémie', thal_label(p.get('thal', 3))),
            ]
            story.append(info_table(params))
            
            # Interprétation clinique
            story.append(Spacer(1, 6*mm))
            if prob > 60:
                interpretation = "⚠️ RISQUE ÉLEVÉ : Une consultation médicale est recommandée dans les plus brefs délais."
            elif prob > 30:
                interpretation = "🟡 RISQUE MODÉRÉ : Une surveillance régulière est conseillée."
            else:
                interpretation = "✅ RISQUE FAIBLE : Continuez vos bonnes habitudes."
            story.append(Paragraph(f"<b>Interprétation clinique :</b> {interpretation}", s['Body']))

        # 2. ANALYSES ECG 

        for i, c in enumerate(ecg_consults):
            story.append(PageBreak())
            
            res = c.get('result', {})
            date_str = c['date'][:10]
            time_str = c['date'][11:16] if len(c['date']) > 10 else ''
            class_name = res.get('class_name', '—')
            confidence = res.get('confidence', 0)
            is_normal = 'Normal' in str(class_name)

            story.append(section_header(f'ANALYSE ECG N°{i+1}', bg=colors.HexColor('#0369a1')))
            story.append(Spacer(1, 4*mm))
            
            story.append(Paragraph(f"<b>Patient :</b> {patient_name}", s['Body']))
            story.append(Paragraph(f"<b>Date de l'analyse :</b> {date_str} à {time_str}", s['Body']))
            story.append(Spacer(1, 6*mm))

            status_color = EMERALD if is_normal else ROSE
            status_text = "ECG NORMAL" if is_normal else f"ANOMALIE DETECTEE : {class_name}"
            
            status_t = Table([[
                Paragraph(f'<font color="white"><b>{status_text}</b></font>',
                          ParagraphStyle('xs', fontName='Helvetica-Bold', fontSize=14,
                                         leading=20, textColor=WHITE, alignment=TA_CENTER))
            ]], colWidths=[16.2*cm])
            status_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), status_color),
                ('TOPPADDING', (0,0), (-1,-1), 12),
                ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(status_t)
            story.append(Spacer(1, 6*mm))

            story.append(info_table([
                ('Classification', class_name),
                ('Confiance', f'{confidence*100:.1f}%'),
                ('Conseil clinique', res.get('clinical_advice', '—')),
            ]))

        # 3. RADIOGRAPHIES 

        for i, c in enumerate(xray_consults):
            story.append(PageBreak())
            
            res = c.get('result', {})
            date_str = c['date'][:10]
            time_str = c['date'][11:16] if len(c['date']) > 10 else ''
            status = res.get('status', '')
            is_normal = status == 'normal'

            story.append(section_header(f'RADIOGRAPHIE N°{i+1}', bg=colors.HexColor('#7c3aed')))
            story.append(Spacer(1, 4*mm))
            
            story.append(Paragraph(f"<b>Patient :</b> {patient_name}", s['Body']))
            story.append(Paragraph(f"<b>Date de l'analyse :</b> {date_str} à {time_str}", s['Body']))
            story.append(Spacer(1, 6*mm))

            status_color = EMERALD if is_normal else ROSE
            status_text = "RADIOGRAPHIE NORMALE" if is_normal else "ANOMALIE DETECTEE"
            
            status_t = Table([[
                Paragraph(f'<font color="white"><b>{status_text}</b></font>',
                          ParagraphStyle('xs', fontName='Helvetica-Bold', fontSize=14,
                                         leading=20, textColor=WHITE, alignment=TA_CENTER))
            ]], colWidths=[16.2*cm])
            status_t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), status_color),
                ('TOPPADDING', (0,0), (-1,-1), 12),
                ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(status_t)
            story.append(Spacer(1, 6*mm))

            scores = res.get('scores', {})
            score_rows = [(k, f'{v}%') for k, v in sorted(scores.items(), key=lambda x: -x[1])[:8]]
            if score_rows:
                story.append(info_table(score_rows))
                story.append(Spacer(1, 4*mm))
                
                # Détection spécifique
                if scores.get('Cardiomegaly', 0) > 50:
                    story.append(Paragraph("<b>⚠️ Cardiomégalie détectée</b> - Une évaluation cardiologique est recommandée.", s['Body']))
                elif scores.get('Effusion', 0) > 50:
                    story.append(Paragraph("<b>⚠️ Épanchement pleural détecté</b> - Une évaluation pulmonaire est recommandée.", s['Body']))

        # CONCLUSION GÉNÉRALE
    
        story.append(PageBreak())
        story.append(section_header('CONCLUSION GENERALE'))
        story.append(Spacer(1, 4*mm))
        
        if risk_consults:
            last_risk = risk_consults[-1]
            prob = last_risk['prediction']['probability'] * 100
            if prob > 60:
                conclusion = f" <b>RISQUE ÉLEVÉ ({prob:.1f}%)</b> - Une consultation médicale est recommandée dans les plus brefs délais."
            elif prob > 30:
                conclusion = f" <b>RISQUE MODÉRÉ ({prob:.1f}%)</b> - Une surveillance régulière est conseillée."
            else:
                conclusion = f" <b>RISQUE FAIBLE ({prob:.1f}%)</b> - Continuez vos bonnes habitudes."
            story.append(Paragraph(conclusion, s['Conclusion']))
        else:
            story.append(Paragraph("Aucune analyse de risque disponible pour établir une conclusion.", s['Conclusion']))

        # Synthèse finale
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("<b>Synthèse des analyses réalisées :</b>", s['Body']))
        story.append(Paragraph(f"• {len(risk_consults)} analyse(s) de risque cardiovasculaire", s['Body']))
        story.append(Paragraph(f"• {len(ecg_consults)} analyse(s) ECG", s['Body']))
        story.append(Paragraph(f"• {len(xray_consults)} radiographie(s) cardiaque(s)", s['Body']))

        story.append(Spacer(1, 8*mm))
        story.append(Paragraph(
            f"Ce rapport a été généré automatiquement par Cardio Digital Twin pour le patient {patient_name}. "
            "Il ne remplace pas un avis médical professionnel.",
            s['Small']
        ))

        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        return filepath