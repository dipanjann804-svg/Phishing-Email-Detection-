"""
Generates a synthetic but realistic sample dataset of phishing and legitimate
emails for demo/training purposes.

Each row also gets randomized fill-in values (amounts, order numbers, names,
domains) so that emails built from the same template are NOT near-duplicates
of each other. This matters because scikit-learn's train_test_split does a
random row split -- if two rows are near-identical (same template, only the
domain swapped), one can end up in train and the other in test, which
inflates reported accuracy in a way that won't hold up on real emails.

NOTE: This is still a synthetic dataset meant to demonstrate the full
pipeline. For a production-grade model, replace data/emails.csv with a real,
larger dataset such as the "Phishing Email Detection" dataset on Kaggle, or
the Nazario / SpamAssassin / Enron corpora.
"""

import csv
import random

random.seed(42)

# ---- Phishing email templates ----
phishing_subjects = [
    "Urgent: Your account will be suspended",
    "Action Required: Verify your account now",
    "Security Alert: Unusual login detected",
    "You have won a prize! Claim now",
    "Your payment failed - update billing info",
    "Immediate action needed: Password expired",
    "Confirm your identity to avoid suspension",
    "Your package could not be delivered",
    "Bank Alert: Unauthorized transaction detected",
    "Final notice: Account closure in 24 hours",
    "Your invoice #{ref} could not be processed",
    "We noticed a new device sign-in",
]

phishing_bodies = [
    "Dear Customer, we detected unusual activity on your account. Click the link below within {hours} hours to verify your identity or your account will be suspended. http://{domain}/verify-account?ref={ref}",
    "Congratulations! You have been selected to receive a ${amount} gift card. Claim your prize now by clicking here: http://{domain}/claim-prize before it expires on request #{ref}.",
    "Your account has been locked due to suspicious activity. Please confirm your username and password at http://{domain}/login-confirm to restore access. Case ID: {ref}",
    "We were unable to process your last payment of ${amount}. Please update your billing details urgently at http://{domain}/billing-update to avoid service interruption.",
    "This is an automated alert from your bank. A transaction of ${amount} was flagged on {date}. Verify this transaction immediately at http://{domain}/secure-verify or your card will be blocked.",
    "Your password will expire in {hours} hours. Click here to reset it now: http://{domain}/reset-password and avoid losing access to your account.",
    "Your parcel (tracking #{ref}) delivery failed due to an incomplete address. Confirm your details within {hours} hours at http://{domain}/delivery-confirm to reschedule delivery.",
    "Dear {name}, our system detected a login attempt from an unrecognized device on {date}. If this wasn't you, secure your account now at http://{domain}/secure-login.",
    "URGENT: Your subscription (order #{ref}) has been cancelled due to a billing issue of ${amount}. Renew immediately at http://{domain}/renew-now to avoid losing your data.",
    "Final Notice: Your account is scheduled for permanent closure on {date}. Verify your information within {hours} hours at http://{domain}/verify-now to prevent this.",
    "Hi {name}, invoice #{ref} for ${amount} could not be processed. Update your payment method at http://{domain}/pay-now to avoid late fees.",
    "We noticed a new device sign-in from an unrecognized location on {date}. If this was not you, click http://{domain}/report-unauthorized immediately.",
]

suspicious_domains = [
    "secure-login-update.com", "account-verify-now.net", "bank-alert-service.info",
    "prize-claim-center.xyz", "billing-update-portal.co", "verify-identity-now.top",
    "login-secure-check.icu", "account-support-team.ru",
]

phishing_sender_names = [
    "Account Security", "Support Team", "Billing Department", "IT Helpdesk", "No-Reply",
]

# ---- Legitimate email templates ----
legit_subjects = [
    "Meeting reminder for tomorrow",
    "Your monthly newsletter",
    "Invoice for your recent purchase",
    "Project update: Q3 milestones",
    "Weekly team standup notes",
    "Your order has shipped",
    "Welcome to our platform",
    "Reminder: Submit your timesheet",
    "Feedback request for recent support ticket",
    "Upcoming maintenance window notice",
    "Re: {ref} follow-up",
    "Your receipt from {domain}",
]

legit_bodies = [
    "Hi team, just a reminder that we have our project sync meeting on {date} at 10 AM in the main conference room. Please bring your status updates.",
    "Hello {name}, thank you for subscribing to our newsletter. Here are this month's highlights from our blog and product updates.",
    "Hi {name}, attached is the invoice #{ref} for your recent purchase of ${amount}. Let us know if you have any questions about the charges or need a receipt.",
    "Hi all, here is a quick update on our Q3 milestones. We are on track to complete the design phase by {date}.",
    "Good morning team, here are the notes from this week's standup on {date}. Please review the action items and update your tickets accordingly.",
    "Hello {name}, good news - your order #{ref} has shipped and should arrive within 3-5 business days. You can track it using the link in your account.",
    "Welcome aboard, {name}! We're excited to have you join our platform. Here is a quick guide to help you get started with your new account.",
    "Hi {name}, this is a friendly reminder to submit your timesheet for this week by end of day Friday so payroll can be processed on time.",
    "Hello, we'd love to hear your feedback on support ticket #{ref} we recently resolved. Your input helps us improve our service.",
    "Hi everyone, please note that our servers will undergo scheduled maintenance on {date} from 1 AM to 3 AM. Some services may be briefly unavailable.",
    "Hi {name}, following up on {ref} - let me know if you have any other questions, happy to help.",
    "Hi {name}, thanks for your order. Your receipt total was ${amount}. Reach out anytime with questions.",
]

legit_domains = [
    "company.com", "ourplatform.io", "teamworkspace.com", "notifications.service.com",
]

legit_sender_names = [
    "Alex Rivera", "Jordan Lee", "Priya Nair", "Support", "Team Updates", "Billing",
]

first_names = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Morgan", "Priya", "Chris"]
dates = ["Monday, June 9", "next Tuesday", "March 14", "this Friday", "August 21", "the 3rd"]


def _rand_ref():
    return str(random.randint(10000, 99999))


def _rand_amount():
    return f"{random.randint(15, 2500):,}.{random.randint(0, 99):02d}"


def _fmt(template, domain):
    return template.format(
        domain=domain,
        ref=_rand_ref(),
        amount=_rand_amount(),
        hours=random.choice([2, 6, 12, 24, 48]),
        date=random.choice(dates),
        name=random.choice(first_names),
    )


def make_phishing_row():
    subject_t = random.choice(phishing_subjects)
    body_t = random.choice(phishing_bodies)
    domain = random.choice(suspicious_domains)
    subject = _fmt(subject_t, domain)
    body = _fmt(body_t, domain)
    sender_name = random.choice(phishing_sender_names)
    sender = f"{sender_name.lower().replace(' ', '.')}@{domain}"
    num_links = body.count("http")
    return subject, body, sender, num_links


def make_legit_row():
    subject_t = random.choice(legit_subjects)
    body_t = random.choice(legit_bodies)
    domain = random.choice(legit_domains)
    subject = _fmt(subject_t, domain)
    body = _fmt(body_t, domain)
    if random.random() < 0.3:
        body += f" More info at https://{domain}/details"
    sender_name = random.choice(legit_sender_names)
    sender = f"{sender_name.lower().replace(' ', '.')}@{domain}"
    num_links = body.count("http")
    return subject, body, sender, num_links


def main(n_per_class=400, out_path="data/emails.csv"):
    rows = []
    for _ in range(n_per_class):
        subject, body, sender, num_links = make_phishing_row()
        rows.append([subject, body, sender, num_links, "phishing"])
    for _ in range(n_per_class):
        subject, body, sender, num_links = make_legit_row()
        rows.append([subject, body, sender, num_links, "legitimate"])

    random.shuffle(rows)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "body", "sender", "num_links", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
