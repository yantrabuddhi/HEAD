define(["application", "./frame", './templates/frames.tpl', 'jquery-ui'],
    function (App, frameView, framesTemplate, emptyTemplate) {
        return Marionette.CompositeView.extend({
            template: framesTemplate,
            childViewContainer: '.app-frames',
            childView: frameView,
            ui: {
                container: '.app-frames'
            },
            initialize: function (options) {
                this.mergeOptions(options, ['layout'])
            },
            childViewOptions: function () {
                return {
                    layout: this.layout
                }
            },
            onRender: function () {
                const self = this;
                this.setOrderNumbers();

                $(this.ui.container).sortable({
                    axis: "y",
                    handle: ".app-frame-drag-handle",
                    placeholder: "ui-state-highlight",
                    deactivate: function () {
                        self.setOrderNumbers();
                    }
                });
            },
            setOrderNumbers: function () {
                const self = this;
                this.ui.container.find('li').each(function (i, frame) {
                    self.children.each(function (motorView) {
                        if (frame == motorView.el) {
                            console.log('set');
                            // updating order_no at relevant model
                            motorView.model.set('order_no', i);
                        }
                    });
                });
            }
        });
    });
