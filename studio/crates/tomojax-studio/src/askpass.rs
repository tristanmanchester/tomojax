//! OpenSSH invokes this small window once per challenge during an explicit login.
use rune::{AppSpec, Cx, El, TextEdit, Tx, col, el, row, text, text_input, tok};
use std::io::Write;

struct Prompt {
    question: String,
    answer: TextEdit,
    confirm: bool,
}
impl Prompt {
    fn new() -> Self {
        let question = std::env::args().nth(1).unwrap_or_default();
        let confirm = std::env::var("SSH_ASKPASS_PROMPT").as_deref() == Ok("confirm")
            || question.contains("(yes/no/");
        Self {
            question,
            answer: TextEdit::new("").secure(),
            confirm,
        }
    }
    fn submit(&mut self, _: &mut Tx) {
        let answer = if self.confirm {
            "yes"
        } else {
            self.answer.text.as_str()
        };
        let mut out = std::io::stdout().lock();
        let sent = writeln!(out, "{answer}").and_then(|()| out.flush()).is_ok();
        std::process::exit(i32::from(!sent))
    }
}
fn view(s: &Prompt, cx: &mut Cx<Prompt>) -> El {
    let t = tok::theme(cx.system_dark());
    let question = text(s.question.clone())
        .font(tok::type_::BODY)
        .color(t.text)
        .wrap();
    let mut body = col()
        .w_full()
        .h_full()
        .padding(18.)
        .gap(12.)
        .fill(t.bg)
        .child(question);
    if !s.confirm {
        body = body.child(
            text_input(&s.answer, tok::type_::BODY)
                .key("answer")
                .label("SSH response")
                .single_line()
                .w_full()
                .h(40.)
                .padding(8.)
                .fill(t.card)
                .color(t.text)
                .on_edit(|s: &mut Prompt| &mut s.answer)
                .on_submit(Prompt::submit),
        );
    }
    body.child(
        row()
            .gap(12.)
            .child(
                el().padding(8.)
                    .fill(t.card)
                    .child(text("Cancel").color(t.text))
                    .on_press(|_: &mut Prompt| std::process::exit(1)),
            )
            .child(
                el().padding(8.)
                    .fill(t.card)
                    .child(text(if s.confirm { "Approve" } else { "Send" }).color(t.text))
                    .on_press_tx(Prompt::submit),
            ),
    )
}

pub fn run() -> ! {
    rune::App::new(AppSpec::new(
        "TomoJAX SSH login",
        (540., 260.),
        Prompt::new,
        view,
    ))
    .run();
    std::process::exit(1)
}
