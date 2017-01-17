define(['backbone', 'path'], function (Backbone, path) {
    return Backbone.Model.extend({
        initialize: function (attributes, options) {
            this.path = options.path || '.';
        },
        urlRoot: function () {
            return path.join('/performance/settings', this.path);
        }
    });
});
